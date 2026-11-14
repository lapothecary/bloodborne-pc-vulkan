// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: objects addressed by small integer ids (images, views, queues). Storage grows in fixed
// blocks, so an object never moves: references to it stay valid while other objects come and go
// (a growing array moved every object, and a reference held across an insert dangled). Freed ids
// are reused last-in first-out; new ones count up from 0.
#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <compare>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "common/types.h"

namespace Common {

struct SlotId {
    static constexpr u32 INVALID_INDEX = std::numeric_limits<u32>::max();

    SlotId() noexcept = default;
    constexpr SlotId(u32 index_) noexcept : index{index_} {}
    constexpr auto operator<=>(const SlotId&) const noexcept = default;
    constexpr explicit operator bool() const noexcept {
        return index != INVALID_INDEX;
    }

    u32 index = INVALID_INDEX;
};

template <class T>
class SlotVector {
    static constexpr u32 BlockShift = 10; ///< 1024 objects per block
    static constexpr u32 BlockSize = 1u << BlockShift;

    struct Block {
        alignas(T) std::byte storage[sizeof(T) * BlockSize];
        u64 used[BlockSize / 64]{};

        T* At(u32 i) {
            return std::launder(reinterpret_cast<T*>(storage) + i);
        }
        const T* At(u32 i) const {
            return std::launder(reinterpret_cast<const T*>(storage) + i);
        }
        bool Used(u32 i) const {
            return (used[i >> 6] >> (i & 63)) & 1;
        }
    };

public:
    template <typename Value, typename Owner>
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::remove_const_t<Value>;
        using difference_type = std::ptrdiff_t;
        using pointer = Value*;
        using reference = Value&;

        Iterator(Owner& owner_, u32 index_) : owner{&owner_}, index{index_} {
            Skip();
        }
        reference operator*() const {
            return (*owner)[SlotId{index}];
        }
        pointer operator->() const {
            return &(*owner)[SlotId{index}];
        }
        Iterator& operator++() {
            ++index;
            Skip();
            return *this;
        }
        Iterator operator++(int) {
            Iterator old = *this;
            ++*this;
            return old;
        }
        bool operator==(const Iterator& other) const {
            return index == other.index;
        }

    private:
        void Skip() {
            while (index < owner->capacity && !owner->is_allocated(SlotId{index})) {
                ++index;
            }
        }
        Owner* owner;
        u32 index;
    };

    using iterator = Iterator<T, SlotVector>;
    using const_iterator = Iterator<const T, const SlotVector>;

    SlotVector() = default;
    SlotVector(const SlotVector&) = delete;
    SlotVector& operator=(const SlotVector&) = delete;

    ~SlotVector() {
        for (auto& block : blocks) {
            for (u32 i = 0; i < BlockSize; ++i) {
                if (block->Used(i)) {
                    std::destroy_at(block->At(i));
                }
            }
        }
    }

    [[nodiscard]] T& operator[](SlotId id) noexcept {
        return *blocks[id.index >> BlockShift]->At(id.index & (BlockSize - 1));
    }
    [[nodiscard]] const T& operator[](SlotId id) const noexcept {
        return *blocks[id.index >> BlockShift]->At(id.index & (BlockSize - 1));
    }

    [[nodiscard]] bool is_allocated(SlotId id) const {
        const u32 block = id.index >> BlockShift;
        return block < blocks.size() && blocks[block]->Used(id.index & (BlockSize - 1));
    }

    /// Phase 4.3d: Wait-free, lock-free lookup of a live slot from `GXWorker:1..5` without racing
    /// on `blocks` `std::vector` growth on Stage B (`Block` storage never moves once allocated).
    [[nodiscard]] const T* try_at_mt(SlotId id) const noexcept {
        const u32 block = id.index >> BlockShift;
        if (block >= MaxMtBlocks) [[unlikely]] {
            return nullptr;
        }
        const Block* blk = mt_blocks[block].load(std::memory_order_acquire);
        if (!blk) [[unlikely]] {
            return nullptr;
        }
        const u32 i = id.index & (BlockSize - 1);
        const u64 mask = __atomic_load_n(&blk->used[i >> 6], __ATOMIC_ACQUIRE);
        if (((mask >> (i & 63)) & 1u) == 0u) [[unlikely]] {
            return nullptr;
        }
        return blk->At(i);
    }

    template <typename... Args>
    SlotId insert(Args&&... args) {
        const u32 index = TakeFreeIndex();
        Block& block = *blocks[index >> BlockShift];
        const u32 i = index & (BlockSize - 1);
        std::construct_at(block.At(i), std::forward<Args>(args)...);
        __atomic_or_fetch(&block.used[i >> 6], u64{1} << (i & 63), __ATOMIC_RELEASE);
        ++live;
        return SlotId{index};
    }

    void erase(SlotId id) {
        Block& block = *blocks[id.index >> BlockShift];
        const u32 i = id.index & (BlockSize - 1);
        __atomic_and_fetch(&block.used[i >> 6], ~(u64{1} << (i & 63)), __ATOMIC_RELEASE);
        std::destroy_at(block.At(i));
        free_ids.push_back(id.index);
        --live;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return live;
    }

    iterator begin() {
        return iterator(*this, 0);
    }
    iterator end() {
        return iterator(*this, capacity);
    }
    const_iterator begin() const {
        return const_iterator(*this, 0);
    }
    const_iterator end() const {
        return const_iterator(*this, capacity);
    }
    const_iterator cbegin() const {
        return begin();
    }
    const_iterator cend() const {
        return end();
    }

private:
    static constexpr u32 MaxMtBlocks = 256; ///< Up to 262,144 slots reachable lock-free on workers

    u32 TakeFreeIndex() {
        if (!free_ids.empty()) {
            const u32 index = free_ids.back();
            free_ids.pop_back();
            return index;
        }
        if (capacity % BlockSize == 0) {
            auto blk = std::make_unique<Block>();
            const u32 bidx = capacity >> BlockShift;
            if (bidx < MaxMtBlocks) {
                mt_blocks[bidx].store(blk.get(), std::memory_order_release);
            }
            blocks.push_back(std::move(blk));
        }
        return capacity++;
    }

    std::array<std::atomic<Block*>, MaxMtBlocks> mt_blocks{};
    std::vector<std::unique_ptr<Block>> blocks;
    std::vector<u32> free_ids;
    u32 capacity = 0; ///< ids handed out so far (free or not)
    std::size_t live = 0;
};

} // namespace Common

template <>
struct std::hash<Common::SlotId> {
    std::size_t operator()(const Common::SlotId& id) const noexcept {
        return std::hash<u32>{}(id.index);
    }
};
