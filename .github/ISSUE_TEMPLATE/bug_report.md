---
name: Bug Report / Сообщение о баге
about: Report a crash, visual glitch, or performance issue / Сообщить о вылете, графическом баге или просадке FPS
title: "[BUG] "
labels: bug
assignees: ''
---

## System Info / Конфигурация ПК
- **OS / ОС:** (e.g. Windows 11 / Windows 10 / Arch Linux / CachyOS / Ubuntu 24.04 / SteamOS)
- **GPU & Driver:** (e.g. RTX 4090 / RTX 3080 / GTX 1050 2 GB / RX 7900 XTX Mesa RADV)
- **CPU:** (e.g. Ryzen 9 7950X / Core i7-13620H)
- **Release Version / Версия релиза:** (e.g. `v1.2.5`)
- **Game Dump Version:** (e.g. `CUSA03173` GOTY v1.09 / `CUSA00900` v1.09)
- **Resolution, Upscaler & Frame Gen:** (e.g. `2560x1440`, Upscaler: `Off` / `DLAA` / `DLSS 4.5` / `FSR 3.1` / `FSR 4` / `XeSS` / `SMAA T2x`, Frame Gen: `Off` / `DLSS-G` / `FSR 3.1`)

## Description / Описание проблемы
Describe what happened and in which location (e.g. Hunter's Dream, Central Yharnam, Clinic).
Опишите, что произошло и в какой локации.

## Optional Isolation Test / Опциональная проверка (если визуальный баг)
- [ ] `BB_NATIVE_SHADERS=0` (falls back from native Vulkan GLSL to translated SPIR-V)
- [ ] `BB_HW_OCCLUSION_CULL=0` (disables 3D hardware occlusion culling)
- [ ] `BB_GX_MODULES=none BB_DIRECT_VULKAN=0 BB_NATIVE_VK_MASK=0` (pure PS4 `eboot.bin` baseline)

## Logs & Screenshots / Логи и скриншоты
Attach the latest log file from `logs/YYYYMMDD_HHMMSS.log` (and `logs/crash-*.log` if it crashed) plus a screenshot if applicable.
Прикрепите последний лог из папки `logs/` (и `logs/crash-*.log` при вылете), а также скриншот.
