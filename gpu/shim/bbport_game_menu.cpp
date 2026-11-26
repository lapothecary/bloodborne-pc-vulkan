// SPDX-License-Identifier: GPL-2.0-or-later
#include "common/types.h"
#include "bbport_game_menu.h"
#include "bbport_fast_load.h"

#include <new>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "bbport_button_prompts.h"
#include "bbport_frame_rate.h"
#include "bbport_guest_hooks.h"
#include "bbport_menu_engine.h"
#include "bbport_settings.h"
#include "bbport_system_info.h"
#include "cloth_exp/cloth_exp_registry.h"
#include "vulkan/low_latency.h"
#include "games/bloodborne_menu_pointer.h"
#include <SDL3/SDL.h>
#include <fmt/format.h>

extern "C" void runtime_file_drain_sync(void);
extern "C" int runtime_savedata_sync_options(const void* options_128);

namespace BbGameMenu {
bool CommitDebouncedDisplaySettings(BbSettings::Values& s, bool force);

namespace {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// The game's menu code (eboot 1.09 vaddrs; found from the System menu builder 0x1bb3ad0).
constexpr u64 MsgLookup = 0xf6f340;      ///< (repository, table, category, id) -> UTF-16 text
constexpr u64 AddPageItem = 0x1b4e2a0;   ///< (menu, name+help texts, page factory, out)
constexpr u64 MakeText = 0x1ae8cc0;      ///< (out 0x40 bytes, category, id): a text entry
constexpr u64 FactoryToFunction = 0x1b6ed90; ///< ({vtable, fn}, out std::function) -> its __f_
constexpr u64 CreatePage = 0x1bb4c70;    ///< (out, arg, content std::function): an options page
constexpr u64 OpenPage = 0x1b20900;      ///< (a, b, layout name, rows fn, 0, 0): the page's content
constexpr u64 ListPush = 0x1b1bbd0;      ///< (value list, {u32 value, text}): up to 32 values
constexpr u64 ListRow = 0x1b78ab0;   ///< (page, texts, int* value, value list, int* default): pop-up list
constexpr u64 PairPush = 0x1b2c200;  ///< (two-value list, {u8 value, text})
constexpr u64 PairRow = 0x1b2a100;   ///< (page, texts, u8* value, two-value list, u8* default): left/right
constexpr u64 SliderRow = 0x1b2ac00; ///< (page, texts, u8* value 0..10, u8* default)
/// (page, texts, int* value, value list, int* default): the value's text in the row, left/right
/// step through the list (the game's own multi-value rows, 0x1b20f74).
constexpr u64 SelectRow = 0x1b29370;
/// A SelectRow row (0xcf0 bytes, vtable 0x5339e90): `int* value` at +0x3c8, its values (0x48 bytes
/// each: u32 value, text) from +0x3d8 (8-aligned), their count at +0xce0, the list widget showing
/// them at +0x100.
constexpr std::size_t RowValue = 0x3c8, RowValues = 0x3d8, RowCount = 0xce0, RowList = 0x100;
/// (row, input, ..., xmm0 frame time): a SelectRow's input, each of the open page's rows every
/// frame (the game's menu thread).
constexpr u64 RowInput = 0x1b26b00;
/// (row + RowList): the row's list widget draws its values' texts (looked up again).
constexpr u64 ListDraw = 0x1ad06e0;
/// (page): updates the bottom StatusBar help text for the page's currently selected row.
constexpr u64 UpdatePageHelpBar = 0x1b1ab20;
/// (page): updates the bottom StatusBar button guide bar for the page's currently selected row.
constexpr u64 UpdatePageButtonGuide = 0x1ad6450;
/// (page, input, frame_time): processes Up/Down row navigation on an options page.
constexpr u64 OptionsPageNavigate = 0x1b1b520;
/// (page): called when the player confirms "Restore to default settings?" -> YES.
/// SelectRow::ResetToDefault (0x1b26f50) unconditionally triggers ListSelect (0x1acf6b0 -> returns 1)
/// and OnListSelect (0x1b27190), which immediately overwrites DefaultResetSentinel (-999) with 0.
/// Detouring this function handles Defaults cleanly for Display, Effects, and Controls pages.
constexpr u64 OptionsPageResetDefaults = 0x1b1a2d0;
/// (page, game_settings): Bloodborne's native System -> Controls page content builder.
constexpr u64 ControlsContent = 0x1b211a0;
/// Menu input button check helpers (const u8* input -> bool / step).
constexpr u64 IsConfirmPressed = 0x1adfec0;
constexpr u64 IsLeftPressed = 0x1ae0890;
constexpr u64 IsRightPressed = 0x1ae0970;
constexpr u64 GetHorizontalStep = 0x1adfdd0;
/// (row): the row shows its value's text (looked up again).
constexpr u64 RowShowValue = 0x1b2b7c0;
/// (out_mc 0x60 bytes, parent_mc 0x60 bytes, fmt, ...): looks up a Scaleform child MovieClip.
constexpr u64 GetSubMovieClip = 0x1adb220;
/// (mc + 0x8, int value): formats an integer into a Scaleform TextField MovieClip.
constexpr u64 SetMovieClipIntText = 0x1ad9e90;
/// Base MovieClipWrapper vtable (+0x10) used when destroying a temporary 0x60-byte MovieClipWrapper.
constexpr u64 MovieClipBaseVtable = 0x539a070;
/// (row + RowList, count): the list widget's number of values.
constexpr u64 ListSetCount = 0x1acf2c0;
/// (row + RowList, index): the list widget moves to a value; (row + RowList): shows where it is.
constexpr u64 ListSelect = 0x1acf6b0, ListShow = 0x1acfb20;
/// The row's copy of its default value (the page's "Default").
constexpr std::size_t RowDefault = 0x3d4;
constexpr u64 FactoryVtable = 0x533d120; ///< a System item's {vtable, page factory fn}
constexpr u64 ContentVtable = 0x5343880; ///< an options page's {vtable, content fn}
/// The option page layout (optionsetting.gfx) the port's pages use, by name: its title's text
/// id and its rows. ControllSetting stacked its six rows upwards, and a row's pop-up list showed
/// the rows below through it: the runtime gives the game a copy with them stacked downwards, as
/// the game's own pop-up rows have them (src/runtime_file.c, menu_layout_file).
struct Layout {
    u64 name;
    u32 title, subtitle; ///< StaticText ids in the layout (0: none)
    std::size_t rows;
};
constexpr Layout ControlsLayout{0x4934065, 113020, 0, 11}; // "ControllSetting" (expanded to 11 rows)
constexpr u32 TextName = 0xc8, TextHelp = 0xc9; // SP_menu text / SP_one-line help
constexpr u32 ControlsItem = 110000;            // "Controls" in the System menu: Keybinds follow it
constexpr u32 ScreenSoundItem = 110001;         // "Screen/Sound" in the System menu: Display/Effects follow it
constexpr u32 TitleNewGameItem = 401303;        // "New Game" / "Новая игра" on the Title Screen
constexpr u64 BindCommandList = 0x1bea820;      // (out_step, menu, cmd_list_mc): binds Title/Startup menu items
constexpr u64 TitleMenuBindRet = 0x1b4adb8;     // return address of BindCommandList inside TitleMenuBuilder (0x1b4a3d0)

const u8 MsgLookupPrologue[] = {0x89, 0xf0, 0x48, 0x8b, 0x77, 0x08, 0x48, 0x8b, 0x34, 0xc6};
const u8 AddPageItemPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};
const u8 RowInputPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};

u64 image_base = 0;
template <typename F>
F Game(u64 va) {
    return reinterpret_cast<F>(image_base + va);
}

// ---- Texts: ids from IdBase up, looked up by the game like its own messages. ----
constexpr u32 IdBase = 0x7f0000;
struct Text {
    std::u16string english, russian;
};
std::vector<Text> texts;

std::u16string Utf16(const char* utf8) {
    std::u16string out;
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(utf8); *p;) {
        u32 c = *p++;
        if (c >= 0xf0) {
            c = (c & 7) << 18 | (p[0] & 0x3f) << 12 | (p[1] & 0x3f) << 6 | (p[2] & 0x3f);
            p += 3;
        } else if (c >= 0xe0) {
            c = (c & 15) << 12 | (p[0] & 0x3f) << 6 | (p[1] & 0x3f);
            p += 2;
        } else if (c >= 0xc0) {
            c = (c & 31) << 6 | (p[0] & 0x3f);
            p += 1;
        }
        if (c >= 0x10000) {
            c -= 0x10000;
            out.push_back(char16_t(0xd800 + (c >> 10)));
            out.push_back(char16_t(0xdc00 + (c & 0x3ff)));
        } else {
            out.push_back(char16_t(c));
        }
    }
    return out;
}
int ActiveOrbisLanguage() {
    static const int lang = [] {
        const char* env = std::getenv("BB_LANGUAGE");
        return (env && *env) ? std::atoi(env) : 1; // 1 = English by default
    }();
    return lang;
}

const char* TranslateForCurrentGameLang(const char* en) {
    const int lang = ActiveOrbisLanguage();
    if (lang == 1 || lang == 18 || lang == 8 || !en) {
        return en;
    }
    struct Entry {
        const char* en;
        const char* fr; // 2
        const char* es; // 3, 20
        const char* de; // 4
        const char* it; // 5
        const char* pt; // 7, 17
        const char* pl; // 16
        const char* ja; // 0
        const char* ko; // 9
        const char* zh; // 10, 11
    };
    static constexpr Entry Table[] = {
        {"Display", "Affichage", "Pantalla", "Anzeige", "Schermo", "Tela", "Ekran", "ディスプレイ", "디스플레이", "显示"},
        {"Effects", "Effets", "Efectos", "Effekte", "Effetti", "Efeitos", "Efekty", "エフェクト", "효과", "特效"},
        {"Aspect Ratio", "Format d'image", "Relación de aspecto", "Seitenverhältnis", "Rapporto d'aspetto", "Proporção de tela", "Proporcje ekranu", "アスペクト比", "화면 비율", "宽高比"},
        {"Resolution", "Résolution", "Resolución", "Auflösung", "Risoluzione", "Resolução", "Rozdzielczość", "解像度", "해상도", "分辨率"},
        {"Window Mode", "Mode d'affichage", "Modo de ventana", "Fenstermodus", "Modalità finestra", "Modo de janela", "Tryb okna", "画面モード", "화면 모드", "窗口模式"},
        {"Fullscreen", "Plein écran", "Pantalla completa", "Vollbild", "Schermo intero", "Tela cheia", "Pełny ekran", "フルスクリーン", "전체 화면", "全屏"},
        {"Borderless Windowed", "Fenêtré sans bordure", "Ventana sin bordes", "Rahmenloses Fenster", "Finestra senza bordi", "Janela sem bordas", "Okno bez ramki", "ボーダレスウィンドウ", "테두리 없는 창", "无边框窗口"},
        {"Windowed", "Fenêtré", "Ventana", "Fenster", "Finestra", "Modo janela", "W oknie", "ウィンドウ", "창 모드", "窗口"},
        {"FPS Limit", "Limite de FPS", "Límite de FPS", "FPS-Limit", "Limite FPS", "Limite de FPS", "Limit FPS", "最大フレームレート", "FPS 제한", "帧率上限"},
        {"Anti-Aliasing", "Anticrénelage", "Antialiasing", "Kantenglättung", "Anti-Aliasing", "Anti-Aliasing", "Wygładzanie krawędzi", "アンチエイリアス", "안티앨리어싱", "抗锯齿"},
        {"Upscaler", "Mise à l'échelle", "Escalador", "Upscaler", "Upscaler", "Upscaler", "Skalowanie", "アップスケーラー", "업스케일러", "超分辨率"},
        {"Upscaler Mode", "Mode d'échelle", "Modo de escalado", "Upscaler-Modus", "Modalità Upscaler", "Modo do Upscaler", "Tryb skalowania", "アップスケールモード", "업스케일 모드", "缩放模式"},
        {"Sharpness (RCAS)", "Netteté (RCAS)", "Nitidez (RCAS)", "Schärfe (RCAS)", "Nitidezza (RCAS)", "Nitidez (RCAS)", "Ostrość (RCAS)", "シャープネス (RCAS)", "선명도 (RCAS)", "锐化 (RCAS)"},
        {"Texture Quality", "Qualité des textures", "Calidad de texturas", "Texturqualität", "Qualità texture", "Qualidade de texturas", "Jakość tekstur", "テクスチャ品質", "텍스처 품질", "纹理质量"},
        {"Shadow Quality", "Qualité des ombres", "Calidad de sombras", "Schattenqualität", "Qualità ombre", "Qualidade das sombras", "Jakość cieni", "シャドウ品質", "그림자 품질", "阴影质量"},
        {"Anisotropic Filtering", "Filtrage anisotrope", "Filtro anisotrópico", "Anisotrope Filterung", "Filtro anisotropico", "Filtro anisotrópico", "Filtrowanie anizotropowe", "異方性フィルタリング", "이방성 필터링", "各向异性过滤"},
        {"Model Detail (LOD)", "Détail des modèles (LOD)", "Detalle de modelos (LOD)", "Modelldetails (LOD)", "Dettaglio modelli (LOD)", "Detalhe dos modelos (LOD)", "Szczegółowość modeli (LOD)", "モデル詳細度 (LOD)", "모델 디테일 (LOD)", "模型细节 (LOD)"},
        {"UI & Icons Upscale", "Upscaling UI & Icônes", "Escalado de UI e iconos", "UI- & Icon-Upscaling", "Upscaling UI e icone", "Upscale de UI e ícones", "Skalowanie UI i ikon", "UI・アイコン高解像度化", "UI 및 아이콘 업스케일", "UI 与图标超分辨率"},
        {"On (AI / SDF 2x-4x)", "Activé (IA / SDF 2x-4x)", "Activado (IA / SDF 2x-4x)", "Ein (KI / SDF 2x-4x)", "Attivato (IA / SDF 2x-4x)", "Ativado (IA / SDF 2x-4x)", "Wł. (AI / SDF 2x-4x)", "オン (AI / SDF 2x-4x)", "켜짐 (AI / SDF 2x-4x)", "开启 (AI / SDF 2x-4x)"},
        {"Uncapped", "Illimité", "Sin límite", "Unbegrenzt", "Illimitato", "Ilimitado", "Bez limitu", "無制限", "제한 없음", "无限制"},
        {"Off", "Désactivé", "Desactivado", "Aus", "Disattivato", "Desativado", "Wył.", "オフ", "꺼짐", "关闭"},
        {"On", "Activé", "Activado", "Ein", "Attivato", "Ativado", "Wł.", "オン", "켜짐", "开启"},
        {"Normal", "Normal", "Normal", "Normal", "Normale", "Normal", "Normalna", "標準", "보통", "正常"},
        {"Inverted", "Inversé", "Invertido", "Invertiert", "Invertito", "Invertido", "Odwrócona", "反転", "반전", "反转"},
        {"Mouse Speed", "Vitesse souris", "Velocidad del ratón", "Mausgeschwindigkeit", "Velocità mouse", "Velocidade do mouse", "Czułość myszy", "マウス感度", "마우스 감도", "鼠标速度"},
        {"Controller Vibration", "Vibrations manette", "Vibración del mando", "Controller-Vibration", "Vibrazione controller", "Vibração do controle", "Wibracje kontrolera", "振動機能", "패드 진동", "手柄震动"},
        {"Camera Sensitivity", "Sensibilité caméra", "Sensibilidad de cámara", "Kameraempfindlichkeit", "Sensibilità telecamera", "Sensibilidade da câmera", "Czułość kamery", "カメラ速度", "카메라 감도", "视角灵敏度"},
        {"Camera X-Axis", "Caméra axe X", "Eje X de cámara", "Kamera X-Achse", "Asse X telecamera", "Eixo X da câmera", "Oś X kamery", "カメラ左右操作", "카메라 좌우 조작", "视角 X 轴"},
        {"Camera Y-Axis", "Caméra axe Y", "Eje Y de cámara", "Kamera Y-Achse", "Asse Y telecamera", "Eixo Y da câmera", "Oś Y kamery", "カメラ上下操作", "카메라 상하 조작", "视角 Y 轴"},
        {"Auto Target Switch", "Changement de cible auto", "Cambio auto. de objetivo", "Auto-Zielwechsel", "Cambio bersaglio auto.", "Troca auto. de alvo", "Auto-zmiana celu", "ターゲット自動変更", "타겟 자동 변경", "自动切换锁定"},
        {"Camera Auto-Adjust", "Ajustement caméra auto", "Ajuste auto. de cámara", "Kamera-Auto-Korrektur", "Regolazione auto. telecamera", "Ajuste auto. da câmera", "Auto-korekta kamery", "カメラ自動補正", "카메라 자동 보정", "视角自动回避"},
        {"Move Forward", "Avancer", "Avanzar", "Vorwärts bewegen", "Muovi avanti", "Mover para frente", "Ruch w przód", "前進", "앞으로 이동", "向前移动"},
        {"Move Back", "Reculer", "Retroceder", "Rückwärts bewegen", "Muovi indietro", "Mover para trás", "Ruch w tył", "後退", "뒤로 이동", "向后移动"},
        {"Move Left", "Aller à gauche", "Mover a la izquierda", "Nach links bewegen", "Muovi a sinistra", "Mover para a esquerda", "Ruch w lewo", "左移動", "왼쪽으로 이동", "向左移动"},
        {"Move Right", "Aller à droite", "Mover a la derecha", "Nach rechts bewegen", "Muovi a destra", "Mover para a direita", "Ruch w prawo", "右移動", "오른쪽으로 이동", "向右移动"},
        {"Dodge / Sprint / Backstep", "Esquive / Sprint / Pas arrière", "Esquivar / Correr / Paso atrás", "Ausweichen / Sprinten / Rückschritt", "Schivata / Scatto / Passo indietro", "Esquiva / Corrida / Passo atrás", "Unik / Sprint / Odskok", "回避 / ダッシュ / バックステップ", "회피 / 질주 / 백스텝", "闪避 / 冲刺 / 后撤步"},
        {"Interact / Confirm", "Interagir / Confirmer", "Interactuar / Confirmar", "Interagieren / Bestätigen", "Interagisci / Conferma", "Interagir / Confirmar", "Interakcja / Potwierdź", "調べる / 決定", "상호작용 / 확인", "互动 / 确认"},
        {"Use Quick Item", "Utiliser objet rapide", "Usar objeto rápido", "Schnellgegenstand benutzen", "Usa oggetto rapido", "Usar item rápido", "Użyj przedmiotu", "アイテム使用", "퀵 아이템 사용", "使用快捷道具"},
        {"Blood Vial (Heal)", "Fiole de sang (Soin)", "Vial de sangre (Curar)", "Blutphiole (Heilen)", "Fiala di sangue (Cura)", "Frasco de sangue (Curar)", "Fiolka krwi (Leczenie)", "輸血液を使用", "수혈액 사용 (회복)", "使用采血瓶（治疗）"},
        {"Light Attack", "Attaque légère", "Ataque ligero", "Leichter Angriff", "Attacco leggero", "Ataque leve", "Szybki atak", "通常攻撃 (R1)", "약공격 (R1)", "轻攻击 (R1)"},
        {"Heavy Attack", "Attaque puissante", "Ataque fuerte", "Schwerer Angriff", "Attacco pesante", "Ataque forte", "Silny atak", "強攻撃 (R2)", "강공격 (R2)", "重攻击 (R2)"},
        {"Firearm Shot / Parry", "Tir d'arme à feu / Parade", "Disparo / Parry", "Schusswaffe / Parieren", "Colpo arma da fuoco / Parry", "Disparo / Parry", "Strzał z broni palnej", "銃撃 / パリィ (L2)", "총격 / 패리 (L2)", "枪击 / 枪反 (L2)"},
        {"Transform Weapon", "Transformer l'arme", "Transformar arma", "Waffe transformieren", "Trasforma arma", "Transformar arma", "Przekształć broń", "武器変形 (L1)", "무기 변형 (L1)", "武器变形 (L1)"},
        {"Lock-On / Reset Camera", "Verrouiller / Réinit. caméra", "Fijar objetivo / Cámara", "Anvisieren / Kamera", "Aggancia / Resetta visuale", "Travar mira / Câmera", "Namierzanie / Kamera", "ターゲット固定 / カメラリセット", "타겟 고정 / 카메라 리셋", "锁定目标 / 重置视角"},
        {"Chromatic Aberration", "Aberration chromatique", "Aberración cromática", "Chromatische Aberration", "Aberrazione cromatica", "Aberração cromática", "Aberracja chromatyczna", "色収差", "색수차", "色差"},
        {"Depth of Field (DoF)", "Profondeur de champ (DoF)", "Profundidad de campo (DoF)", "Tärfenschärfe (DoF)", "Profondità di campo (DoF)", "Profundidade de campo (DoF)", "Głębia ostrości (DoF)", "被写界深度 (DoF)", "피사계 심도 (DoF)", "景深 (DoF)"},
        {"Motion Blur", "Flou de mouvement", "Desenfoque de movimiento", "Bewegungsunschärfe", "Motion Blur", "Desfoque de movimento", "Rozmycie w ruchu", "モーションブラー", "모션 블러", "动态模糊"},
        {"Ambient Occlusion (SSAO)", "Occlusion ambiante (SSAO)", "Oclusión ambiental (SSAO)", "Umgebungsverdeckung (SSAO)", "Occlusione ambientale (SSAO)", "Oclusão ambiental (SSAO)", "Okluzja otoczenia (SSAO)", "アンビエントオクルージョン (SSAO)", "앰비언트 오클루전 (SSAO)", "环境光遮蔽 (SSAO)"},
        {"Dynamic Shadows", "Ombres dynamiques", "Sombras dinámicas", "Dynamische Schatten", "Ombre dinamiche", "Sombras dinâmicas", "Dynamiczne cienie", "ダイナミックシャドウ", "동적 그림자", "动态光源阴影"},
        {"SSR Reflections", "Reflets SSR", "Reflejos SSR", "SSR-Reflexionen", "Riflessi SSR", "Reflexos SSR", "Odbicia SSR", "スクリーンスペース反射 (SSR)", "화면 공간 반사 (SSR)", "屏幕空间反射 (SSR)"},
        {"Bloom", "Bloom (Lueur)", "Bloom (Resplandor)", "Bloom (Überstrahlen)", "Bloom (Bagliore)", "Bloom (Brilho)", "Poświata (Bloom)", "ブルーム (光彩)", "블룸 (광채)", "泛光 (Bloom)"},
        {"Enhanced GTAO / SSAO", "GTAO / SSAO amélioré", "GTAO / SSAO mejorado", "Verbessertes GTAO / SSAO", "GTAO / SSAO migliorato", "GTAO / SSAO aprimorado", "Ulepszone GTAO / SSAO", "高精度 GTAO / SSAO", "향상된 GTAO / SSAO", "增强型双尺度 GTAO / SSAO"},
        {"Contact Micro-Shadows", "Micro-ombres de contact", "Micro-sombras de contacto", "Kontakt-Mikroschatten", "Micro-ombre di contatto", "Micro-sombras de contato", "Kontaktowe mikro-cienie", "コンタクトマイクロシャドウ", "접촉 마이크로 그림자", "屏幕空间接触微阴影"},
        {"High-Precision Ultra SSR", "Ultra SSR haute précision", "Ultra SSR de alta precisión", "Hochpräzises Ultra-SSR", "Ultra SSR ad alta precisione", "Ultra SSR de alta precisão", "Wysokoprecyzyjne Ultra SSR", "高精度ウルトラ SSR", "고정밀 울트라 SSR", "高精度 Ultra SSR (2x 步进)"},
        {"True Black & Vibrance", "Noir profond (OLED) & Vibrance", "Negro puro (OLED) y viveza", "Tiefschwarz (OLED) & Brillanz", "Nero puro (OLED) e vividezza", "Preto puro (OLED) e vivacidade", "Głęboka czerń (OLED) i nasycenie", "トゥルーブラック (OLED) & 彩度", "트루 블랙 (OLED) 및 채도", "纯黑去雾 (OLED) 与色彩鲜艳度"},
        {"Native HDR10 (PQ)", "HDR10 natif (PQ)", "HDR10 nativo (PQ)", "Natives HDR10 (PQ)", "HDR10 nativo (PQ)", "HDR10 nativo (PQ)", "Natywny HDR10 (PQ)", "ネイティブ HDR10 (PQ)", "네이티브 HDR10 (PQ)", "原生 HDR10 输出 (PQ)"},
        {"Debug", "Debug", "Depuración", "Debug", "Debug", "Debug", "Debug", "デバッグ", "디버그", "调试信息"},
        {"Additional Debug", "Debug détaillé", "Depuración detallada", "Erweitertes Debug", "Debug dettagliato", "Debug detalhado", "Szczegółowy debug", "詳細デバッグ", "상세 디버그", "详细调试信息"},
        {"Camera Follow Hotkey", "Raccourci suivi caméra", "Atajo seguimiento cámara", "Kamera-Folgemodus-Taste", "Tasto modalità telecamera", "Atalho modo de câmera", "Skrót trybu kamery", "カメラ追従切替キー", "카메라 추적 전환 키", "视角跟随切换热键"},
        {"Camera Follow Mode", "Suivi de la caméra", "Seguimiento de cámara", "Kamera-Folgemodus", "Inseguimento telecamera", "Seguimento da câmera", "Śledzenie kamery", "カメラ追従モード", "카메라 추적 모드", "视角跟随模式"},
        {"Loading Screen Tips Timer", "Minuteur d'astuces", "Temporizador de consejos", "Ladebildschirm-Tipps-Timer", "Timer consigli caricamento", "Timer de dicas na carga", "Timer wskazówek ładowania", "ロード画面ヒント表示時間", "로딩 화면 팁 타이머", "加载界面提示计时器"},
        {"Original (PS4)", "Original (PS4)", "Original (PS4)", "Original (PS4)", "Originale (PS4)", "Original (PS4)", "Oryginalny (PS4)", "オリジナル (PS4)", "오리지널 (PS4)", "原版惯性 (PS4)"},
        {"Balanced", "Équilibré", "Equilibrado", "Ausgewogen", "Bilanciato", "Equilibrado", "Zbalansowany", "バランス", "밸런스", "均衡"},
        {"Locked (Rigid 1:1)", "Verrouillé (Rigide 1:1)", "Fijo (Rígido 1:1)", "Fixiert (Starr 1:1)", "Bloccato (Rigido 1:1)", "Fixo (Rígido 1:1)", "Sztywny (1:1)", "固定 (リジッド 1:1)", "고정 (1:1 직결)", "完全锁定 (1:1 刚性)"},
        {"Exit Game", "Quitter le jeu", "Salir del juego", "Spiel beenden", "Esci dal gioco", "Sair do jogo", "Wyjdź z gry", "ゲームを終了", "게임 종료", "退出游戏"},
        {"Exit Bloodborne and return to desktop.", "Quitter Bloodborne et revenir au bureau.", "Salir de Bloodborne y volver al escritorio.", "Bloodborne beenden und zum Desktop zurückkehren.", "Esci da Bloodborne e torna al desktop.", "Sair de Bloodborne e voltar para a área de trabalho.", "Zamknij Bloodborne i wróć do pulpitu.", "Bloodborneを終了してデスクトップに戻ります。", "Bloodborne을 종료하고 바탕 화면으로 돌아갑니다.", "退出《血源诅咒》并返回桌面。"},
        {"Default", "Par défaut", "Predeterminado", "Standard", "Predefinito", "Padrão", "Domyślne", "標準", "기본값", "默认"},
    };
    for (const auto& e : Table) {
        if (std::strcmp(en, e.en) == 0) {
            switch (lang) {
                case 2: return e.fr;
                case 3:
                case 20: return e.es;
                case 4: return e.de;
                case 5: return e.it;
                case 7:
                case 17: return e.pt;
                case 16: return e.pl;
                case 0: return e.ja;
                case 9: return e.ko;
                case 10:
                case 11: return e.zh;
                default: return en;
            }
        }
    }
    return en;
}

u32 AddText(const char* english, const char* russian) {
    texts.push_back({Utf16(TranslateForCurrentGameLang(english)), Utf16(russian)});
    return IdBase + u32(texts.size() - 1);
}
/// The game's own language decides (its "New Game" / "Screen/Sound" text in Cyrillic: Russian,
/// else English); falls back to BB_LANGUAGE=8 or BB_MENU_LANGUAGE=ru before any FMG lookup.
enum class GameLanguage { Unknown, English, Russian };
std::atomic<GameLanguage> game_language{GameLanguage::Unknown};

bool RussianTexts() {
    const GameLanguage gl = game_language.load(std::memory_order_relaxed);
    if (gl == GameLanguage::Russian || ActiveOrbisLanguage() == 8) {
        return true;
    }
    if (gl == GameLanguage::English) {
        return false;
    }
    const char* menu_lang = std::getenv("BB_MENU_LANGUAGE");
    return menu_lang && (menu_lang[0] == 'r' || menu_lang[0] == 'R') &&
           (menu_lang[1] == 'u' || menu_lang[1] == 'U');
}

// Texts that change while the game runs (the chosen aspect ratio's resolutions, "Auto (21:9)",
// the 6 dynamic rows inside the unified Controls page, and the 6 dynamic rows inside Display):
// fixed buffers rewritten in place, so a text the game holds stays valid.
constexpr u32 DynamicBase = IdBase + 0x8000;
constexpr std::size_t DynamicLength = 128;
struct DynamicText {
    char16_t english[DynamicLength]{}, russian[DynamicLength]{};
};
constexpr u32 OutputSlots = 16;    ///< BbSettings::OutputList
constexpr int ControlsSlotCount = 11; ///< all 11 visible rows in Controls scroll across all 35 items
constexpr int FirstControlSettingItem = BbSettings::KeyActionCount;
constexpr int ItemMouseSpeed = FirstControlSettingItem + 0;
constexpr int ItemVibration = FirstControlSettingItem + 1;
constexpr int ItemCameraSens = FirstControlSettingItem + 2;
constexpr int ItemCameraFollow = FirstControlSettingItem + 3;
constexpr int ItemCameraX = FirstControlSettingItem + 4;
constexpr int ItemCameraY = FirstControlSettingItem + 5;
constexpr int ItemAutoTarget = FirstControlSettingItem + 6;
constexpr int ItemCameraAdjust = FirstControlSettingItem + 7;
constexpr int ItemLoadingTipsTimer = FirstControlSettingItem + 8;
constexpr int TotalControlsItems = BbSettings::KeyActionCount + 9; ///< 27 keybinds + Mouse Speed + Camera Follow + Loading Tips Timer + 6 native settings
constexpr int MaxControlsScroll = TotalControlsItems - ControlsSlotCount;
constexpr int DisplaySlotCount = 11; ///< 11 visible rows per screen in Display
enum DisplayItem : int {
    DispItemAspect = 0,
    DispItemResolution,
    DispItemWindowMode,
    DispItemFpsLimit,
    DispItemVsync,
    DispItemLowLatency,
    DispItemAntiAliasing,
    DispItemUpscaler,
    DispItemUpscalerMode,
    DispItemSharpness,
    DispItemFrameGeneration,
    DispItemTextureQuality,
    DispItemShadowQuality,
    DispItemAnisotropic,
    DispItemModelLod,
    DispItemUiUpscale,
    DispItemClothPhysics,
    TotalDisplayItems
};
constexpr int MaxDisplayScroll = TotalDisplayItems - DisplaySlotCount;
constexpr int EffectsSlotCount = 11; ///< all 11 Effects settings fit on a single screen (0 scroll)
constexpr int TotalEffectsItems = 16;
constexpr int MaxEffectsScroll = TotalEffectsItems - EffectsSlotCount;
constexpr int MaxChoicesPerSlot = 24;
constexpr int DefaultResetSentinel = -999;
std::array<DynamicText, 1 + OutputSlots + 3 * ControlsSlotCount + 3 * DisplaySlotCount + 3 * EffectsSlotCount +
                        ControlsSlotCount * 4 + DisplaySlotCount * MaxChoicesPerSlot + EffectsSlotCount * 4>
    dynamic_texts{};
constexpr u32 AutoText = DynamicBase;                                       ///< "Auto (21:9)"
constexpr u32 OutputTexts = DynamicBase + 1;                                ///< the resolutions (OutputSlots)
constexpr u32 ControlsSlotName = OutputTexts + OutputSlots;                 ///< 6 dynamic row labels in Controls
constexpr u32 ControlsSlotHelp = ControlsSlotName + ControlsSlotCount;      ///< 6 dynamic row helps in Controls
constexpr u32 ControlsSlotVal = ControlsSlotHelp + ControlsSlotCount;       ///< 6 dynamic row values in Controls
constexpr u32 DisplaySlotName = ControlsSlotVal + ControlsSlotCount;        ///< 6 dynamic row labels in Display
constexpr u32 DisplaySlotHelp = DisplaySlotName + DisplaySlotCount;         ///< 6 dynamic row helps in Display
constexpr u32 DisplaySlotVal = DisplaySlotHelp + DisplaySlotCount;          ///< 6 dynamic row values in Display
constexpr u32 ControlsSlotChoices = DisplaySlotVal + DisplaySlotCount;      ///< 6 * 4 choice texts in Controls
constexpr u32 DisplaySlotChoices = ControlsSlotChoices + ControlsSlotCount * 4; ///< 6 * 16 choice texts in Display
constexpr u32 EffectsSlotName = DisplaySlotChoices + DisplaySlotCount * MaxChoicesPerSlot; ///< 6 dynamic row labels in Effects
constexpr u32 EffectsSlotHelp = EffectsSlotName + EffectsSlotCount;         ///< 6 dynamic row helps in Effects
constexpr u32 EffectsSlotVal = EffectsSlotHelp + EffectsSlotCount;          ///< 6 dynamic row values in Effects
constexpr u32 EffectsSlotChoices = EffectsSlotVal + EffectsSlotCount;       ///< 6 * 4 choice texts in Effects

void Store16(char16_t (&out)[DynamicLength], const char* utf8) {
    const std::u16string text = Utf16(utf8);
    const std::size_t n = std::min(text.size(), DynamicLength - 1);
    for (std::size_t i = 0; i < n; ++i) {
        __atomic_store_n(&out[i], text[i], __ATOMIC_RELAXED);
    }
    __atomic_store_n(&out[n], char16_t(0), __ATOMIC_RELAXED);
}
void SetDynamic(u32 id, const char* english, const char* russian) {
    DynamicText& t = dynamic_texts[id - DynamicBase];
    Store16(t.english, TranslateForCurrentGameLang(english));
    Store16(t.russian, russian);
}

bool OurText(u32 id) {
    return (id >= IdBase && id < IdBase + texts.size()) ||
           (id >= DynamicBase && id < DynamicBase + dynamic_texts.size());
}
const char16_t* TextOf(u32 id) {
    const bool russian = RussianTexts();
    if (id >= DynamicBase) {
        const DynamicText& t = dynamic_texts[id - DynamicBase];
        return russian ? t.russian : t.english;
    }
    const Text& t = texts[id - IdBase];
    return russian ? t.russian.c_str() : t.english.c_str();
}

// ---- The port's values the game's rows edit (ints: the list rows bind ints). ----
enum Field : int {
    Aspect, OutputRes, Upscaler, Preset, Sharpness, ModelLod,
    FirstEffect,
    FirstSlot = FirstEffect + BbSettings::EffectCount,
    FirstDisplaySlot = FirstSlot + ControlsSlotCount,
    FirstEffectsSlot = FirstDisplaySlot + DisplaySlotCount,
    FieldCount = int(FirstEffectsSlot) + EffectsSlotCount
};
int values[FieldCount];  // what the game's rows show and change (list rows bind ints)
u8 bytes[FieldCount];    // the same for toggle and slider rows (they bind bytes)
bool byte_field[FieldCount]; // fields shown by toggle or slider rows
int default_values[FieldCount]; // the port's defaults: what the page's "Default" sets
u8 default_bytes[FieldCount];
int applied[FieldCount]; // what Poll applied last
std::atomic<bool> values_valid{false};
constexpr float SharpnessStep = 0.2f; // the game's slider: 0..10 -> 0.0 .. 2.0
constexpr int SharpnessSteps = 11;
constexpr float MouseSensStep = 0.2f; // 0 off, 1..10 -> 0.2 .. 2.0
constexpr int MouseSensSteps = 11;
constexpr int LodValues[] = {-2, 0, 1, 2};
/// The resolution row's values: the outputs of the aspect ratio it shows (value i: entry i). Set
/// when the page is built and when another aspect ratio is chosen on it (RefreshOutputRow, the
/// game's menu thread; Poll reads it once the rows have rested).
BbSettings::OutputList menu_outputs;
int output_row_aspect = -1; ///< the aspect ratio the resolution row shows

BbSettings::OutputSize MenuOutput(int aspect, int index) {
    if (menu_outputs.count == 0) return BbSettings::NearestOutput(aspect, BbSettings::OutputDefault);
    return BbSettings::NearestOutput(aspect, menu_outputs.sizes[std::clamp(index, 0, menu_outputs.count - 1)]);
}

/// The resolution row's value of `output` (0 if none).
int MenuOutputIndex(int aspect, BbSettings::OutputSize output) {
    for (int i = 0; i < menu_outputs.count; ++i) {
        if (MenuOutput(aspect, i) == output) return i;
    }
    return 0;
}

/// The resolution row's texts for `aspect`, "Auto (..)" for the monitor's aspect ratio.
void RefreshOutputs(int aspect) {
    const std::string group = BbSettings::AspectKey(BbSettings::ResolvedAspect(BbSettings::AspectAuto));
    SetDynamic(AutoText, ("Auto (" + group + ")").c_str(), ("Авто (" + group + ")").c_str());
    for (int i = 0; i < int(OutputSlots); ++i) {
        const auto o = MenuOutput(aspect, i);
        const std::string text = std::to_string(o.width) + " x " + std::to_string(o.height);
        SetDynamic(OutputTexts + u32(i), text.c_str(), text.c_str());
    }
}

void SyncDisplayStateFromSettings();

/// The rows' values of settings `s`.
void Fill(const BbSettings::Values& s, int* values) {
    values[Aspect] = s.aspect;
    values[OutputRes] = MenuOutputIndex(s.aspect, BbSettings::NearestOutput(
                                                      s.aspect, BbSettings::UnpackSize(s.output_res)));
    values[Upscaler] = s.upscaler;
    values[Preset] = s.preset;
    // One slider for both: 0 is sharpening off.
    values[Sharpness] =
        s.sharpen ? std::clamp(int(std::lround(s.sharpness / SharpnessStep)), 1, SharpnessSteps - 1) : 0;
    values[ModelLod] = 1;
    for (int i = 0; i < 4; ++i) {
        if (LodValues[i] == s.model_lod) values[ModelLod] = i;
    }
    for (int e = 0; e < BbSettings::EffectCount; ++e) {
        values[FirstEffect + e] = s.effects[e];
    }
    for (int slot = 0; slot < ControlsSlotCount; ++slot) {
        values[FirstSlot + slot] = 0;
    }
    for (int slot = 0; slot < DisplaySlotCount; ++slot) {
        values[FirstDisplaySlot + slot] = 0;
    }
    for (int slot = 0; slot < EffectsSlotCount; ++slot) {
        values[FirstEffectsSlot + slot] = 0;
    }
}

void LoadValues() {
    const int aspect = BbSettings::Get().aspect;
    menu_outputs = BbSettings::Outputs(aspect);
    RefreshOutputs(aspect);
    Fill(BbSettings::Get(), values);
    SyncDisplayStateFromSettings();
    output_row_aspect = values[Aspect];
    for (int f = 0; f < FieldCount; ++f) {
        bytes[f] = u8(values[f]);
    }
    // "Default" on the page: the game's resolution in this list.
    default_values[OutputRes] =
        MenuOutputIndex(aspect, BbSettings::NearestOutput(aspect, BbSettings::OutputDefault));
    std::memcpy(applied, values, sizeof(values));
    values_valid = true;
}

// ---- Pages and rows ----
enum class Kind { List, Toggle, Slider, Select };
struct Row {
    Field field;
    u32 name, help;
    std::vector<u32> choices; ///< value texts, value = index (a toggle: off, on)
    Kind kind = Kind::List;
};
struct Page {
    u32 name, help;
    const Layout* layout;
    std::vector<Row> rows; ///< at most the layout's
};
std::vector<Page> pages;

std::vector<Row> controls_rows;
u32 exit_game_name = 0;
u32 exit_game_help = 0;

void DefineTexts() {
    exit_game_name = AddText("Exit Game", "Выйти из игры");
    exit_game_help = AddText("Exit Bloodborne and return to desktop.",
                             "Выйти из Bloodborne на рабочий стол.");
    const u32 off = AddText("Off", "Выкл"), on = AddText("On", "Вкл");
    const std::vector<u32> toggle = {off, on};
    const auto page = [](const char* en, const char* ru, const char* help_en, const char* help_ru,
                         const Layout& layout) {
        const u32 name = AddText(en, ru);
        pages.push_back({name, AddText(help_en, help_ru), &layout, {}});
    };
    const auto row = [](Field f, const char* en, const char* ru, const char* help_en,
                        const char* help_ru, std::vector<u32> choices, Kind kind = Kind::List) {
        if (kind == Kind::List && choices.size() == 2) {
            kind = Kind::Toggle; // left/right in place, as the game's own on/off rows
        }
        pages.back().rows.push_back(
            {f, AddText(en, ru), AddText(help_en, help_ru), std::move(choices), kind});
    };

    // Scrollable 9-item Display page (Aspect Ratio, Resolution, FPS Limit, Anti-Aliasing,
    // Upscaler, Upscaler Mode, Sharpness, Anisotropic Filtering, Model Detail):
    page("Display", "Изображение",
         "Resolution, FPS limit, Anti-Aliasing (DLAA/TAA/FXAA), Upscaler (DLSS 4/FSR) & Filtering",
         "Разрешение, лимит FPS, сглаживание (DLAA/TAA/FXAA), апскейлер (DLSS 4/FSR) и фильтрация",
         ControlsLayout);
    for (int slot = 0; slot < DisplaySlotCount; ++slot) {
        pages.back().rows.push_back(
            {Field(FirstDisplaySlot + slot),
             DisplaySlotName + u32(slot),
             DisplaySlotHelp + u32(slot),
             {DisplaySlotVal + u32(slot), DisplaySlotVal + u32(slot)},
             Kind::Select});
    }

    page("Effects", "Эффекты",
         "Visual effects & Vulkan 1.3 Remaster Enhancements (all 11 switch live in-game)",
         "Визуальные эффекты и улучшения ремастера Vulkan 1.3 (все 11 меняются на лету)",
         ControlsLayout);
    for (int slot = 0; slot < EffectsSlotCount; ++slot) {
        pages.back().rows.push_back(
            {Field(FirstEffectsSlot + slot),
             EffectsSlotName + u32(slot),
             EffectsSlotHelp + u32(slot),
             {EffectsSlotVal + u32(slot), EffectsSlotVal + u32(slot)},
             Kind::Select});
    }

    // Unified System -> Controls window: all 6 visible rows scroll vertically across all 20 items
    // (13 Keybinds + Mouse Speed + Controller Vibration + Camera Sensitivity + 4 Camera toggles).
    for (int slot = 0; slot < ControlsSlotCount; ++slot) {
        controls_rows.push_back(
            {Field(FirstSlot + slot),
             ControlsSlotName + u32(slot),
             ControlsSlotHelp + u32(slot),
             {ControlsSlotVal + u32(slot), ControlsSlotVal + u32(slot)},
             Kind::Select});
    }
}

// ---- The game's structures, as its own code builds them. ----
constexpr std::size_t TextSize = 0x40;     // text entry (MakeText): text pointer, string, flag
constexpr std::size_t ValueSize = 0x48;    // {u32 value; text}
constexpr std::size_t ListSize = 0x920;    // 32 values + count at +0x908 (+ alignment slack)
constexpr std::size_t PairListSize = 0xb0; // 2 values + count at +0x98 (+ alignment slack)
constexpr std::size_t ListCountOffset = 0x908, PairCountOffset = 0x98;
constexpr std::size_t FunctionSize = 0x28; // std::function: 32-byte buffer, __f_ at +0x20

/// Destroys a std::function the way the game's code does (its __f_ at +0x20).
void DestroyFunction(u8* f) {
    void* impl = *reinterpret_cast<void**>(f + 0x20);
    if (!impl) {
        return;
    }
    using Destroy = void (PS4_SYSV_ABI*)(void*, int);
    Destroy destroy = (*reinterpret_cast<Destroy**>(impl))[4];
    destroy(impl, impl != f);
    *reinterpret_cast<void**>(f + 0x20) = nullptr;
}

/// Frees a text entry's string the way the game's code does (heap buffer once capacity >= 8).
void DestroyText(u8* text) {
    if (*reinterpret_cast<u64*>(text + 0x28) >= 8) {
        void* allocator = *reinterpret_cast<void**>(text + 0x30);
        using Free = void (PS4_SYSV_ABI*)(void*, void*);
        (*reinterpret_cast<Free**>(allocator))[0x70 / 8](allocator, *reinterpret_cast<void**>(text + 0x10));
    }
    *reinterpret_cast<u64*>(text + 0x28) = 7;
    *reinterpret_cast<u64*>(text + 0x20) = 0;
    *reinterpret_cast<char16_t*>(text + 0x10) = 0;
}
void DestroyTexts(u8* pair) {
    DestroyText(pair);
    DestroyText(pair + TextSize);
}
/// A value list's entries (count at `count_offset`), after the row has copied them.
void DestroyValues(u8* list, std::size_t count_offset) {
    const u64 count = *reinterpret_cast<u64*>(list + count_offset);
    for (u64 i = 0; i < count; ++i) {
        DestroyText(list + i * ValueSize + 8);
    }
}

void MakeTexts(u8* pair, u32 name, u32 help) {
    using Make = void* (PS4_SYSV_ABI*)(void*, u32, u32);
    Game<Make>(MakeText)(pair, TextName, name);
    Game<Make>(MakeText)(pair + TextSize, TextHelp, help);
}

/// Upscalers this GPU and build can run (known once the device exists, before any menu).
bool UpscalerAvailable(int i) {
    const auto& s = BbSettings::Get();
    return i == BbSettings::UpscalerFsr4     ? s.fsr4_supported.load()
           : i == BbSettings::UpscalerFsr411 ? s.fsr411_supported.load()
           : i == BbSettings::UpscalerDlss   ? s.dlss_supported.load()
                                             : true;
}

/// Another aspect ratio chosen on the page: the resolution row gets that aspect ratio's outputs
/// as its values (as many; the texts are looked up in the port's buffers, RefreshOutputs, when the
/// row shows them) and keeps the chosen height.
void RefreshOutputRow(u8* row) {
    const int aspect = std::clamp(values[Aspect], 0, BbSettings::AspectCount - 1);
    if (aspect == output_row_aspect) return;
    const BbSettings::OutputSize shown = MenuOutput(output_row_aspect, values[OutputRes]);
    output_row_aspect = aspect;
    menu_outputs = BbSettings::Outputs(aspect);
    RefreshOutputs(aspect);
    u8* items = row + RowValues;
    items += (0 - reinterpret_cast<std::uintptr_t>(items)) & 7;
    u64& count = *reinterpret_cast<u64*>(row + RowCount);
    // More values: copies of the first (its text an empty in-place string, see SelectRow) with
    // their own value and text.
    const u64 want = u64(menu_outputs.count);
    const bool in_place = *reinterpret_cast<u64*>(items + 0x30) < 8;
    if (count > 0 && want <= 32 && (want <= count || in_place)) {
        for (u64 i = count; i < want; ++i) {
            std::memcpy(items + i * ValueSize, items, ValueSize);
        }
        for (u64 i = 0; i < want; ++i) {
            *reinterpret_cast<u32*>(items + i * ValueSize) = u32(i);
            *reinterpret_cast<const char16_t**>(items + i * ValueSize + 8) = TextOf(OutputTexts + u32(i));
        }
        count = want;
        Game<void (PS4_SYSV_ABI*)(void*, u32)>(ListSetCount)(row + RowList, u32(want));
    }
    const int value = MenuOutputIndex(aspect, BbSettings::NearestOutput(aspect, shown));
    values[OutputRes] = value;
    default_values[OutputRes] = MenuOutputIndex(aspect, BbSettings::NearestOutput(aspect, BbSettings::OutputDefault));
    *reinterpret_cast<int*>(row + RowDefault) = default_values[OutputRes];
    Game<int (PS4_SYSV_ABI*)(void*, u32)>(ListSelect)(row + RowList, u32(value));
    Game<void (PS4_SYSV_ABI*)(void*)>(ListShow)(row + RowList);
    Game<void (PS4_SYSV_ABI*)(void*)>(RowShowValue)(row);
    Game<void (PS4_SYSV_ABI*)(void*)>(ListDraw)(row + RowList);
}

void AddRows(void* page, const std::vector<Row>& rows) {
    using Push = void (PS4_SYSV_ABI*)(void*, void*);
    using AddRow = void (PS4_SYSV_ABI*)(void*, void*, int*, void*, const int*);
    using Make = void* (PS4_SYSV_ABI*)(void*, u32, u32);
    for (const Row& row : rows) {
        alignas(16) u8 pair[2 * TextSize] = {};
        MakeTexts(pair, row.name, row.help);
        if (row.kind == Kind::Slider) {
            using Slider = void (PS4_SYSV_ABI*)(void*, void*, u8*, const u8*);
            Game<Slider>(SliderRow)(page, pair, &bytes[row.field], &default_bytes[row.field]);
            DestroyTexts(pair);
            continue;
        }
        if (row.kind == Kind::Toggle) {
            using PairRowFn = void (PS4_SYSV_ABI*)(void*, void*, u8*, void*, const u8*);
            u8* two = static_cast<u8*>(::operator new(PairListSize, std::align_val_t{16}));
            std::memset(two, 0, PairListSize);
            for (std::size_t i = 0; i < 2; ++i) {
                alignas(16) u8 value[ValueSize] = {};
                value[0] = u8(i);
                Game<Make>(MakeText)(value + 8, TextName, row.choices[i]);
                Game<Push>(PairPush)(two, value);
                DestroyText(value + 8);
            }
            Game<PairRowFn>(PairRow)(page, pair, &bytes[row.field], two, &default_bytes[row.field]);
            DestroyValues(two, PairCountOffset);
            ::operator delete(two, std::align_val_t{16});
            DestroyTexts(pair);
            continue;
        }
        u8* list = static_cast<u8*>(::operator new(ListSize, std::align_val_t{16}));
        std::memset(list, 0, ListSize);
        // The resolution row: as many as the aspect ratio has (its texts are rewritten).
        const std::size_t count =
            row.field == OutputRes ? std::size_t(menu_outputs.count) : row.choices.size();
        for (std::size_t i = 0; i < count; ++i) {
            if (row.field == Upscaler && !UpscalerAvailable(int(i))) {
                continue; // the list's values are the upscaler numbers, not positions
            }
            alignas(16) u8 value[ValueSize] = {};
            *reinterpret_cast<u32*>(value) = u32(i);
            Game<Make>(MakeText)(value + 8, TextName, row.choices[i]);
            Game<Push>(ListPush)(list, value);
            DestroyText(value + 8);
        }
        Game<AddRow>(row.kind == Kind::Select ? SelectRow : ListRow)(page, pair, &values[row.field], list,
                                                                    &default_values[row.field]);
        DestroyValues(list, ListCountOffset);
        ::operator delete(list, std::align_val_t{16});
        DestroyTexts(pair);
    }
}

std::atomic<int> building_page{-1}; // while the game builds one of our pages (its menu thread)
void InitDisplayPage(void* page);
void InitEffectsPage(void* page);

// Called by the game's menu code (guest threads, System V ABI like the game), one set per page.
template <int P>
void PS4_SYSV_ABI Rows(void* page, void*) {
    if constexpr (P == 0) {
        InitDisplayPage(page);
    } else if constexpr (P == 1) {
        InitEffectsPage(page);
    } else {
        LoadValues();
        AddRows(page, pages[P].rows);
    }
}
template <int P>
void* PS4_SYSV_ABI Content(void* a, void* b) {
    using Open = void* (PS4_SYSV_ABI*)(void*, void*, const char*, void*, u64, u64);
    void* page = Game<Open>(OpenPage)(a, b, Game<const char*>(pages[P].layout->name),
                                      reinterpret_cast<void*>(&Rows<P>), 0, 0);
    building_page = -1; // the layout's texts were looked up
    return page;
}
void* Factory(void* out, void* arg, void* content, int page) {
    using Create = void* (PS4_SYSV_ABI*)(void*, void*, void*);
    alignas(16) u8 function[FunctionSize] = {};
    *reinterpret_cast<u64*>(function) = image_base + ContentVtable;
    *reinterpret_cast<void**>(function + 8) = content;
    *reinterpret_cast<void**>(function + 0x20) = function;
    // The page is built later (the game's menu task): its layout title is looked up right before
    // the content function runs, which takes the override back.
    building_page = page;
    Game<Create>(CreatePage)(out, arg, function);
    DestroyFunction(function);
    return out;
}
template <int P>
void* PS4_SYSV_ABI PageFactory(void* out, void* arg) {
    return Factory(out, arg, reinterpret_cast<void*>(&Content<P>), P);
}
constexpr std::size_t MaxPages = 6;
void* const factories[MaxPages] = {
    reinterpret_cast<void*>(&PageFactory<0>), reinterpret_cast<void*>(&PageFactory<1>),
    reinterpret_cast<void*>(&PageFactory<2>), reinterpret_cast<void*>(&PageFactory<3>),
    reinterpret_cast<void*>(&PageFactory<4>), reinterpret_cast<void*>(&PageFactory<5>)};

// ---- Unified System -> Controls Window (Keybinds + Mouse + Camera/Controller) ----
enum class CapturePhase { WaitInitialRelease, WaitPressAndRelease, WaitEscapeRelease };
std::atomic<int> capturing_action{-1};
std::atomic<int> captured_wheel{0};
int capturing_slot = -1;
bool capturing_alt = false; ///< false: Primary Key 1, true: Alt Key 2
CapturePhase capture_phase = CapturePhase::WaitInitialRelease;
std::vector<std::string> captured_tokens;
std::string last_capture_preview;
std::atomic<long long> capture_cooldown_ms{0};
std::atomic<long long> last_capture_step_ms{0};

u8* active_controls_page = nullptr;
std::atomic<bool> in_controls_page{false};
u8* active_game_settings = nullptr;
u8* slot_rows[ControlsSlotCount] = {};
int controls_scroll = 0;
bool controls_wrapped_from_top = false;

u8* active_display_page = nullptr;
u8* display_slot_rows[DisplaySlotCount] = {};
int display_scroll = 0;
bool display_wrapped_from_top = false;

u8* active_effects_page = nullptr;
u8* effects_slot_rows[EffectsSlotCount] = {};
int effects_scroll = 0;
bool effects_wrapped_from_top = false;

void ClearActiveCustomPages() {
    active_controls_page = nullptr;
    active_display_page = nullptr;
    active_effects_page = nullptr;
    in_controls_page.store(false, std::memory_order_relaxed);
    BbPrompts::SetControlsGuideActive(false);
    for (int i = 0; i < ControlsSlotCount; ++i) {
        slot_rows[i] = nullptr;
    }
    for (int i = 0; i < DisplaySlotCount; ++i) {
        display_slot_rows[i] = nullptr;
    }
    for (int i = 0; i < EffectsSlotCount; ++i) {
        effects_slot_rows[i] = nullptr;
    }
}

bool ValidateActivePageRows(u8* page, int slot_count, const int* first_val_ptr, u8** rows) {
    if (!page) {
        return false;
    }
    u8* const rows_begin = *reinterpret_cast<u8**>(page + 0xe58);
    u8* const rows_end = *reinterpret_cast<u8**>(page + 0xe60);
    if (!rows_begin || !rows_end || (rows_end - rows_begin) < slot_count * 0x90) {
        return false;
    }
    for (int slot = 0; slot < slot_count; ++slot) {
        u8* const widget = *reinterpret_cast<u8**>(rows_begin + slot * 0x90 + 0x80);
        if (!widget) {
            return false;
        }
        const int* const val_ptr = *reinterpret_cast<const int* const*>(widget + RowValue);
        if (val_ptr != (first_val_ptr + slot)) {
            return false;
        }
        rows[slot] = widget;
    }
    return true;
}

int GetSelectedPageRow(u8* page) {
    if (!page) return 0;
    int sel = *reinterpret_cast<int*>(page + 0xae8);
    if (sel < 0) sel = *reinterpret_cast<int*>(page + 0xae4);
    return sel;
}

int GetSelectedControlsRow() {
    return GetSelectedPageRow(active_controls_page);
}

int GetSelectedDisplayRow() {
    return GetSelectedPageRow(active_display_page);
}

int GetSelectedEffectsRow() {
    return GetSelectedPageRow(active_effects_page);
}

long long NowSteadyMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool IsModifierToken(const std::string& tok) {
    return tok == "Left Shift" || tok == "Right Shift" || tok == "Left Ctrl" ||
           tok == "Right Ctrl" || tok == "Left Alt" || tok == "Right Alt";
}

void SplitBinding(const std::string& raw, std::string& primary, std::string& alt) {
    const std::size_t comma = raw.find(',');
    if (comma == std::string::npos) {
        primary = raw;
        alt.clear();
    } else {
        primary = raw.substr(0, comma);
        alt = raw.substr(comma + 1);
    }
}

std::string JoinBinding(const std::string& primary, const std::string& alt) {
    if (primary.empty()) return alt;
    if (alt.empty() || alt == primary) return primary;
    return primary + "," + alt;
}

std::vector<std::string> CollectDownTokens(bool* escape_down) {
    std::vector<std::string> out;
    if (escape_down) *escape_down = false;
    if (!SDL_WasInit(SDL_INIT_VIDEO)) return out;
    int num_keys = 0;
    const bool* keys = SDL_GetKeyboardState(&num_keys);
    if (keys) {
        if (num_keys > SDL_SCANCODE_ESCAPE && keys[SDL_SCANCODE_ESCAPE]) {
            if (escape_down) *escape_down = true;
        }
        for (int sc = 4; sc < std::min(num_keys, int(SDL_SCANCODE_COUNT)); ++sc) {
            if (sc == SDL_SCANCODE_ESCAPE || !keys[sc]) continue;
            const char* raw_name = SDL_GetScancodeName(SDL_Scancode(sc));
            if (!raw_name || !*raw_name) continue;
            std::string name = raw_name;
            if (name == "Return") name = "Enter";
            out.push_back(std::move(name));
        }
    }
    const SDL_MouseButtonFlags mb = SDL_GetMouseState(nullptr, nullptr);
    if (mb & SDL_BUTTON_LMASK) out.emplace_back("Mouse Left");
    if (mb & SDL_BUTTON_RMASK) out.emplace_back("Mouse Right");
    if (mb & SDL_BUTTON_MMASK) out.emplace_back("Mouse Middle");
    if (mb & SDL_BUTTON_X1MASK) out.emplace_back("Mouse 4");
    if (mb & SDL_BUTTON_X2MASK) out.emplace_back("Mouse 5");
    return out;
}

void PlayMenuClickSound() {
    if (*reinterpret_cast<int*>(image_base + 0x557fc64) <= 0) {
        *reinterpret_cast<u64*>(image_base + 0x557fc60) = 0x3e8;
        *reinterpret_cast<u64*>(image_base + 0x557fc68) = image_base + 0x4933d56;
    }
}

void PlayMenuSliderSound() {
    if (*reinterpret_cast<int*>(image_base + 0x557fc64) <= 0) {
        *reinterpret_cast<u64*>(image_base + 0x557fc60) = 0x190;
        *reinterpret_cast<u64*>(image_base + 0x557fc68) = image_base + 0x493387b;
    }
}

void PlayMenuStepSound() {
    if (*reinterpret_cast<int*>(image_base + 0x557fc64) <= 0) {
        *reinterpret_cast<u64*>(image_base + 0x557fc60) = 0x0;
        *reinterpret_cast<u64*>(image_base + 0x557fc68) = image_base + 0x49344e4;
    }
}

/// Returns keybind action index (0..12) for item_idx (0..19), or -1 if it's a slider/toggle setting.
int ItemKeyAction(int item_idx) {
    return (item_idx >= 0 && item_idx < BbSettings::KeyActionCount) ? item_idx : -1;
}

int CurrentMouseSpeedStep() {
    const auto& s = BbSettings::Get();
    return s.mouse_look
               ? std::clamp(int(std::lround(s.mouse_sensitivity / MouseSensStep)), 1, MouseSensSteps - 1)
               : 0;
}

void SetMouseSpeedStep(int step) {
    auto& s = BbSettings::Get();
    step = std::clamp(step, 0, MouseSensSteps - 1);
    s.mouse_look = step != 0;
    if (step != 0) {
        s.mouse_sensitivity = step * MouseSensStep;
    }
    BbSettings::Save();
}

void RefreshSlotTexts(int slot) {
    const int item_idx = std::clamp(controls_scroll + slot, 0, TotalControlsItems - 1);
    const u32 name_id = ControlsSlotName + u32(slot);
    const u32 help_id = ControlsSlotHelp + u32(slot);
    const u32 val_id = ControlsSlotVal + u32(slot);

    const int k = ItemKeyAction(item_idx);
    if (k >= 0) {
        const auto& def = BbSettings::GetKeyActionDef(k);
        SetDynamic(name_id, def.name_en, def.name_ru);
        SetDynamic(help_id, def.help_en, def.help_ru);
        const std::string raw = BbSettings::GetKeyBinding(k);
        const std::string en = BbSettings::FormatKeyBinding(raw, false);
        const std::string ru = BbSettings::FormatKeyBinding(raw, true);
        SetDynamic(val_id, en.c_str(), ru.c_str());
        return;
    }

    if (item_idx == ItemMouseSpeed) {
        SetDynamic(name_id, "Mouse Speed", "Чувствит. мыши");
        SetDynamic(help_id,
                   "Mouse camera sensitivity (Left/Right): 0 mouse look off, 5 default (1.0x), 10 fast (2.0x)",
                   "Чувствительность камеры мыши (Влево/Вправо): 0 выкл, 5 обычная (1.0x), 10 быстрая (2.0x)");
        const int step = CurrentMouseSpeedStep();
        const std::string txt = fmt::format("{}", step);
        SetDynamic(val_id, txt.c_str(), txt.c_str());
        return;
    }

    if (item_idx == ItemVibration) {
        SetDynamic(name_id, "Controller Vibration", "Вибрация геймпада");
        SetDynamic(help_id,
                   "Controller vibration strength (Left/Right): 0 off, 10 strongest",
                   "Сила вибрации геймпада (Влево/Вправо): 0 выкл, 10 макс.");
        const int vib = active_game_settings ? std::clamp(int(active_game_settings[0x1]), 0, 10) : 10;
        const std::string txt = fmt::format("{}", vib);
        SetDynamic(val_id, txt.c_str(), txt.c_str());
        return;
    }

    if (item_idx == ItemCameraSens) {
        SetDynamic(name_id, "Camera Sensitivity", "Скорость камеры");
        SetDynamic(help_id, "Gamepad & lock-on camera rotation speed (Left/Right: 0 .. 10)",
                   "Скорость вращения камеры (Влево/Вправо: 0 .. 10)");
        const int sens = active_game_settings ? std::clamp(int(active_game_settings[0x0]), 0, 10) : 5;
        const std::string txt = fmt::format("{}", sens);
        SetDynamic(val_id, txt.c_str(), txt.c_str());
        return;
    }

    if (item_idx == ItemCameraFollow) {
        SetDynamic(name_id, "Camera Follow Mode", "Следование камеры");
        SetDynamic(help_id,
                   "Camera follow inertia: Original (PS4), Balanced (spring + subtle sprint pull-back), or Locked (1:1). Hotkey: F4",
                   "Плавность следования камеры: Оригинал (PS4), Баланс (пружина + плавный спринт) или Жёсткая (1:1). Клавиша: F4");
        const int mode = std::clamp(
            BbSettings::Get().camera_follow_mode.load(std::memory_order_relaxed),
            0, BbSettings::CameraFollowCount - 1);
        const char* en = mode == BbSettings::CameraFollowOriginal   ? "Original (PS4)"
                         : mode == BbSettings::CameraFollowBalanced ? "Balanced"
                                                                    : "Locked (Rigid 1:1)";
        const char* ru = mode == BbSettings::CameraFollowOriginal   ? "Оригинал (PS4)"
                         : mode == BbSettings::CameraFollowBalanced ? "Баланс"
                                                                    : "Жёсткая (1:1)";
        SetDynamic(val_id, en, ru);
        return;
    }

    if (item_idx == ItemCameraX) {
        SetDynamic(name_id, "Camera X-Axis", "Камера по оси X");
        SetDynamic(help_id, "Horizontal camera rotation direction (Left/Right to toggle)",
                   "Направление вращения камеры по горизонтали (Влево/Вправо)");
        const bool inv = active_game_settings && active_game_settings[0x9] == 0;
        SetDynamic(val_id, inv ? "Inverted" : "Normal", inv ? "Инверсия" : "Обычная");
    } else if (item_idx == ItemCameraY) {
        SetDynamic(name_id, "Camera Y-Axis", "Камера по оси Y");
        SetDynamic(help_id, "Vertical camera rotation direction (Left/Right to toggle)",
                   "Направление вращения камеры по вертикали (Влево/Вправо)");
        const bool inv = active_game_settings && active_game_settings[0xa] == 0;
        SetDynamic(val_id, inv ? "Inverted" : "Normal", inv ? "Инверсия" : "Обычная");
    } else if (item_idx == ItemAutoTarget) {
        SetDynamic(name_id, "Auto Target Switch", "Автосмена цели");
        SetDynamic(help_id, "Automatically switch lock-on target when current target is defeated",
                   "Автоматическое переключение захвата цели после гибели текущей цели");
        const bool on = !active_game_settings || active_game_settings[0xb] != 0;
        SetDynamic(val_id, on ? "On" : "Off", on ? "Вкл" : "Выкл");
    } else if (item_idx == ItemCameraAdjust) {
        SetDynamic(name_id, "Camera Auto-Adjust", "Автокоррекция камеры");
        SetDynamic(help_id, "Automatically adjust camera when colliding with walls",
                   "Автоматический обход стен при движении камеры");
        const bool on = !active_game_settings || active_game_settings[0xc] != 0;
        SetDynamic(val_id, on ? "On" : "Off", on ? "Вкл" : "Выкл");
    } else if (item_idx == ItemLoadingTipsTimer) {
        SetDynamic(name_id, "Loading Screen Tips Timer", "Таймер подсказок загрузки");
        SetDynamic(help_id,
                   "Keep original 12-second minimum timer to read item tips on loading screens (Off = instant close)",
                   "Оставить оригинальный 12-сек таймер для чтения описаний на экране загрузки (Выкл = мгновенно)");
        const bool on = BbSettings::Get().loading_tips_timer.load(std::memory_order_relaxed);
        SetDynamic(val_id, on ? "On" : "Off", on ? "Вкл" : "Выкл");
    }
}

int GetItemSliderValue(int item_idx) {
    if (item_idx == ItemMouseSpeed) {
        return CurrentMouseSpeedStep();
    }
    if (item_idx == ItemVibration) {
        return active_game_settings ? std::clamp(int(active_game_settings[0x1]), 0, 10) : 10;
    }
    if (item_idx == ItemCameraSens) {
        return active_game_settings ? std::clamp(int(active_game_settings[0x0]), 0, 10) : 5;
    }
    return -1;
}

void SetMovieClipVisible(u8* mc, int visible) {
    BbMenuEngine::SetVisible(mc, visible != 0);
}

void DestroyMovieClip(u8* mc) {
    BbMenuEngine::DestroyClip(mc, image_base);
}

void SetMovieClipFrameLabel(u8* mc, const char* label) {
    BbMenuEngine::GotoAndStopLabel(mc, label);
}

void SetMovieClipWideText(u8* text_mc, const char16_t* str) {
    BbMenuEngine::SetWideText(text_mc, str);
}

void SyncPageSlotSliderWidget(u8* page, u8* row, int slot, int slider_val, bool is_focused,
                              const char16_t* combo_text = nullptr) {
    if (!row || !page) return;
    using GetSubFn = void (PS4_SYSV_ABI*)(void*, const void*, const char*, ...);
    auto get_sub = Game<GetSubFn>(GetSubMovieClip);

    alignas(16) u8 item_mc[0x60] = {};
    alignas(16) u8 slider_mc[0x60] = {};
    alignas(16) u8 combo_mc[0x60] = {};
    get_sub(item_mc, page + 0xdf0, "Item_%d_0", slot);
    get_sub(slider_mc, item_mc, "Widgets/Slider");
    get_sub(combo_mc, item_mc, "Widgets/ComboBox");

    if (slider_val >= 0) {
        SetMovieClipVisible(row + 0x10, 0);
        SetMovieClipVisible(combo_mc, 0);
        SetMovieClipVisible(slider_mc, 1);

        alignas(16) u8 cursor_mc[0x60] = {};
        get_sub(cursor_mc, slider_mc, "Cursor");
        SetMovieClipVisible(cursor_mc, is_focused ? 1 : 0);
        DestroyMovieClip(cursor_mc);

        alignas(16) u8 left_mc[0x60] = {};
        alignas(16) u8 right_mc[0x60] = {};
        get_sub(left_mc, slider_mc, "LeftArrow");
        get_sub(right_mc, slider_mc, "RightArrow");
        SetMovieClipVisible(left_mc, (is_focused && slider_val > 0) ? 1 : 0);
        SetMovieClipVisible(right_mc, (is_focused && slider_val < 10) ? 1 : 0);
        DestroyMovieClip(right_mc);
        DestroyMovieClip(left_mc);

        alignas(16) u8 text_mc[0x60] = {};
        get_sub(text_mc, slider_mc, "Text_0");
        Game<void (PS4_SYSV_ABI*)(void*, int)>(SetMovieClipIntText)(text_mc + 0x8, slider_val);
        DestroyMovieClip(text_mc);

        alignas(16) u8 bar_mc[0x60] = {};
        get_sub(bar_mc, slider_mc, "Bar");
        BbMenuEngine::GotoAndStopFrame(bar_mc, u64(1 + std::clamp(slider_val, 0, 10) * 10));
        DestroyMovieClip(bar_mc);
    } else if (combo_text != nullptr) {
        SetMovieClipVisible(row + 0x10, 0);
        SetMovieClipVisible(slider_mc, 0);
        SetMovieClipVisible(combo_mc, 1);

        alignas(16) u8 item0_mc[0x60] = {};
        get_sub(item0_mc, combo_mc, "Item_%d_%d", 0, 0);
        SetMovieClipVisible(item0_mc, 1);

        alignas(16) u8 text_mc[0x60] = {};
        get_sub(text_mc, item0_mc, "Text_0");
        SetMovieClipWideText(text_mc, combo_text);
        DestroyMovieClip(text_mc);
        DestroyMovieClip(item0_mc);

        // In `optionsetting.gfx`, `Widgets/ComboBox` (Sprite 57) has `Cursor` (Sprite 56) as a
        // direct child of `ComboBox`, containing both the golden glow (id=49) and the `<|` / `|>`
        // arrows (id=55 at depths 4 and 6).
        alignas(16) u8 cursor_mc[0x60] = {};
        get_sub(cursor_mc, combo_mc, "Cursor");
        SetMovieClipVisible(cursor_mc, is_focused ? 1 : 0);
        DestroyMovieClip(cursor_mc);
    } else {
        SetMovieClipVisible(slider_mc, 0);
        SetMovieClipVisible(combo_mc, 0);
        SetMovieClipVisible(row + 0x10, 1);
    }

    DestroyMovieClip(combo_mc);
    DestroyMovieClip(slider_mc);
    DestroyMovieClip(item_mc);
}

void SyncSlotSliderWidget(u8* row, int slot, bool is_focused) {
    const int item_idx = std::clamp(controls_scroll + slot, 0, TotalControlsItems - 1);
    const char16_t* combo_text =
        (item_idx >= ItemCameraX) ? TextOf(ControlsSlotVal + u32(slot)) : nullptr;
    SyncPageSlotSliderWidget(active_controls_page, row, slot, GetItemSliderValue(item_idx), is_focused,
                             combo_text);
}

void PopulateSelectRowChoices(u8* row, int field_idx, const std::vector<u32>& choice_ids, int selected_idx) {
    if (!row || choice_ids.empty()) return;
    u8* items = row + RowValues;
    items += (0 - reinterpret_cast<std::uintptr_t>(items)) & 7;
    u64& count = *reinterpret_cast<u64*>(row + RowCount);
    const u64 want = std::min<u64>(choice_ids.size(), 32);
    const bool in_place = *reinterpret_cast<u64*>(items + 0x30) < 8;
    if (count > 0 && want <= 32 && (want <= count || in_place)) {
        for (u64 i = count; i < want; ++i) {
            std::memcpy(items + i * ValueSize, items, ValueSize);
        }
        for (u64 i = 0; i < want; ++i) {
            *reinterpret_cast<u32*>(items + i * ValueSize) = u32(i);
            *reinterpret_cast<const char16_t**>(items + i * ValueSize + 8) = TextOf(choice_ids[i]);
        }
        count = want;
        Game<void (PS4_SYSV_ABI*)(void*, u32)>(ListSetCount)(row + RowList, u32(want));
    }
    const int clamped = std::clamp(selected_idx, 0, int(want) - 1);
    values[field_idx] = clamped;
    applied[field_idx] = clamped;
    default_values[field_idx] = DefaultResetSentinel;
    *reinterpret_cast<int*>(row + RowDefault) = DefaultResetSentinel;
    Game<int (PS4_SYSV_ABI*)(void*, u32)>(ListSelect)(row + RowList, u32(clamped));
    Game<void (PS4_SYSV_ABI*)(void*)>(ListShow)(row + RowList);
    Game<void (PS4_SYSV_ABI*)(void*)>(RowShowValue)(row);
    Game<void (PS4_SYSV_ABI*)(void*)>(ListDraw)(row + RowList);
}

void BuildControlsSlotChoices(int slot, std::vector<u32>& out_choices, int& out_selected_idx) {
    out_choices.clear();
    out_selected_idx = 0;
    const int item_idx = std::clamp(controls_scroll + slot, 0, TotalControlsItems - 1);
    const u32 base_id = ControlsSlotChoices + u32(slot * 4);
    auto add_choice = [&](const char* en, const char* ru) {
        const u32 id = base_id + u32(out_choices.size());
        SetDynamic(id, en, ru);
        out_choices.push_back(id);
    };
    if (item_idx == ItemCameraFollow) {
        add_choice("Original (PS4)", "Оригинал (PS4)");
        add_choice("Balanced", "Баланс");
        add_choice("Locked (Rigid 1:1)", "Жёсткая (1:1)");
        out_selected_idx = std::clamp(
            BbSettings::Get().camera_follow_mode.load(std::memory_order_relaxed),
            0, BbSettings::CameraFollowCount - 1);
    } else if (item_idx == ItemCameraX) {
        add_choice("Normal", "Обычная");
        add_choice("Inverted", "Инверсия");
        out_selected_idx = (active_game_settings && active_game_settings[0x9] == 0) ? 1 : 0;
    } else if (item_idx == ItemCameraY) {
        add_choice("Normal", "Обычная");
        add_choice("Inverted", "Инверсия");
        out_selected_idx = (active_game_settings && active_game_settings[0xa] == 0) ? 1 : 0;
    } else if (item_idx == ItemAutoTarget) {
        add_choice("Off", "Выкл");
        add_choice("On", "Вкл");
        out_selected_idx = (!active_game_settings || active_game_settings[0xb] != 0) ? 1 : 0;
    } else if (item_idx == ItemCameraAdjust) {
        add_choice("Off", "Выкл");
        add_choice("On", "Вкл");
        out_selected_idx = (!active_game_settings || active_game_settings[0xc] != 0) ? 1 : 0;
    } else if (item_idx == ItemLoadingTipsTimer) {
        add_choice("Off", "Выкл");
        add_choice("On", "Вкл");
        out_selected_idx =
            BbSettings::Get().loading_tips_timer.load(std::memory_order_relaxed) ? 1 : 0;
    }
}

void UpdateSlotRowDisplay(u8* row, int slot) {
    if (!row) return;
    const int item_idx = std::clamp(controls_scroll + slot, 0, TotalControlsItems - 1);
    if (item_idx >= ItemCameraFollow) {
        std::vector<u32> choice_ids;
        int selected_idx = 0;
        BuildControlsSlotChoices(slot, choice_ids, selected_idx);
        PopulateSelectRowChoices(row, FirstSlot + slot, choice_ids, selected_idx);
    } else {
        const std::vector<u32> single = {ControlsSlotVal + u32(slot), ControlsSlotVal + u32(slot)};
        PopulateSelectRowChoices(row, FirstSlot + slot, single, 0);
    }
    const int sel = GetSelectedControlsRow();
    SyncSlotSliderWidget(row, slot, slot == sel);
}

void RefreshControlsPage() {
    controls_scroll = std::clamp(controls_scroll, 0, MaxControlsScroll);
    for (int slot = 0; slot < ControlsSlotCount; ++slot) {
        RefreshSlotTexts(slot);
    }
    if (!ValidateActivePageRows(active_controls_page, ControlsSlotCount, &values[FirstSlot], slot_rows)) {
        return;
    }
    u8* const rows_begin = *reinterpret_cast<u8**>(active_controls_page + 0xe58);
    for (int slot = 0; slot < ControlsSlotCount; ++slot) {
        u8* const entry = rows_begin + slot * 0x90;
        *reinterpret_cast<const char16_t**>(entry + 0x00) = TextOf(ControlsSlotName + u32(slot));
        *reinterpret_cast<const char16_t**>(entry + 0x40) = TextOf(ControlsSlotHelp + u32(slot));
    }
    Game<void (PS4_SYSV_ABI*)(void*)>(ListDraw)(active_controls_page + 0xa70);
    Game<void (PS4_SYSV_ABI*)(void*)>(UpdatePageHelpBar)(active_controls_page);
    Game<void (PS4_SYSV_ABI*)(void*)>(UpdatePageButtonGuide)(active_controls_page);
    for (int slot = 0; slot < ControlsSlotCount; ++slot) {
        UpdateSlotRowDisplay(slot_rows[slot], slot);
    }
    BbMenuEngine::SyncPortScrollBar(
        active_controls_page + BbMenuEngine::OptionPageLayout::kRootMovieClip, image_base,
        controls_scroll, ControlsSlotCount, TotalControlsItems);
}

// ---- Unified Native CSOptionsData <-> bbport.ini <-> SPRJ0005/userdata0010 Sync ----
std::mutex native_options_mutex;
BbMenuEngine::CSOptionsData pending_native_options{};
std::atomic<bool> native_options_dirty{false};
std::atomic<long long> native_options_dirty_ms{0};
std::atomic<bool> native_options_bootstrapped{false};
std::atomic<long long> last_external_ini_poll_ms{0};

BbMenuEngine::CSOptionsData* ActiveSystemOptions() {
    if (!image_base) return nullptr;
    u8* const gdm = *reinterpret_cast<u8* const*>(image_base + BbMenuEngine::Vma::GameDataManPtr);
    if (!gdm) return nullptr;
    return *reinterpret_cast<BbMenuEngine::CSOptionsData**>(gdm + 0x48);
}

void RequestNativeSaveFlag() {
    if (!image_base) return;
    u8* const slm = *reinterpret_cast<u8* const*>(image_base + BbMenuEngine::Vma::FrpgSaveLoadManPtr);
    if (!slm) return;
    if (slm[BbMenuEngine::FrpgSaveLoadManLayout::kSaveEnabled] != 0) {
        slm[BbMenuEngine::FrpgSaveLoadManLayout::kRequestSaveSystemArg] = 0;
        slm[BbMenuEngine::FrpgSaveLoadManLayout::kRequestSaveSystemData] = 1;
    }
}

std::string PasswordToUtf8(const char16_t pw[9]) {
    std::string out;
    for (int i = 0; i < 8 && pw[i] != 0; ++i) {
        const u32 c = u32(pw[i]);
        if (c < 0x80) {
            out.push_back(char(c));
        } else if (c < 0x800) {
            out.push_back(char(0xc0 | (c >> 6)));
            out.push_back(char(0x80 | (c & 0x3f)));
        } else {
            out.push_back(char(0xe0 | (c >> 12)));
            out.push_back(char(0x80 | ((c >> 6) & 0x3f)));
            out.push_back(char(0x80 | (c & 0x3f)));
        }
    }
    return out;
}

void Utf8ToPassword(const std::string& utf8, char16_t out[9]) {
    std::memset(out, 0, sizeof(char16_t) * 9);
    const std::u16string wide = Utf16(utf8.c_str());
    const std::size_t n = std::min<std::size_t>(wide.size(), 8);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = wide[i];
    }
    out[8] = 0;
}

void QueueNativeOptionsDiskSync(const BbMenuEngine::CSOptionsData& opt) {
    {
        std::scoped_lock lock{native_options_mutex};
        pending_native_options = opt;
    }
    native_options_dirty_ms.store(NowSteadyMs(), std::memory_order_relaxed);
    native_options_dirty.store(true, std::memory_order_release);
}

bool FlushNativeOptionsDiskSync(bool wait_if_busy) {
    if (!native_options_dirty.load(std::memory_order_acquire)) {
        return true;
    }
    BbMenuEngine::CSOptionsData snap{};
    if (const auto* live = ActiveSystemOptions()) {
        snap = *live;
    } else {
        std::scoped_lock lock{native_options_mutex};
        snap = pending_native_options;
    }
    const int tries = wait_if_busy ? 40 : 1;
    for (int attempt = 0; attempt < tries; ++attempt) {
        if (runtime_savedata_sync_options(&snap)) {
            native_options_dirty.store(false, std::memory_order_release);
            return true;
        }
        if (wait_if_busy) {
            SDL_Delay(10);
        }
    }
    return false;
}

constexpr u32 kAllNonLangNativeBits =
    BbSettings::Values::NativeBgmVolume | BbSettings::Values::NativeSeVolume |
    BbSettings::Values::NativeVoiceVolume | BbSettings::Values::NativeBloodExpression |
    BbSettings::Values::NativeSubtitles | BbSettings::Values::NativeGameBrightness |
    BbSettings::Values::NativeCameraSpeed | BbSettings::Values::NativePadVibration |
    BbSettings::Values::NativeCameraInvertX | BbSettings::Values::NativeCameraInvertY |
    BbSettings::Values::NativeAutoTarget | BbSettings::Values::NativeCameraAutoAdjust |
    BbSettings::Values::NativeNetworkRegion | BbSettings::Values::NativePlayerNameDisplay |
    BbSettings::Values::NativeNetworkPassword;

/// Reconciles bbport.ini and CSOptionsData on startup / save load:
/// keys already present in bbport.ini override userdata0010; keys absent on first upgrade adopt userdata0010.
void ReconcileNativeOptionsOnLoad(BbMenuEngine::CSOptionsData* opt) {
    if (!opt) return;
    auto& s = BbSettings::Get();
    const u32 mask = s.native_chosen_mask.load(std::memory_order_relaxed);
    bool modified_opt = false;

    auto sync_u8 = [&](u32 bit, std::atomic<int>& setting, u8& field, int lo, int hi) {
        if (mask & bit) {
            const u8 wanted = u8(std::clamp(setting.load(std::memory_order_relaxed), lo, hi));
            if (field != wanted) {
                field = wanted;
                modified_opt = true;
            }
        } else {
            setting.store(std::clamp(int(field), lo, hi), std::memory_order_relaxed);
        }
    };
    auto sync_bool = [&](u32 bit, std::atomic<bool>& setting, u8& field, bool inverted) {
        if (mask & bit) {
            const bool b = setting.load(std::memory_order_relaxed);
            const u8 wanted = inverted ? (b ? 0 : 1) : (b ? 1 : 0);
            if (field != wanted) {
                field = wanted;
                modified_opt = true;
            }
        } else {
            setting.store(inverted ? (field == 0) : (field != 0), std::memory_order_relaxed);
        }
    };

    sync_u8(BbSettings::Values::NativeBgmVolume, s.bgm_volume, opt->music_volume, 0, 10);
    sync_u8(BbSettings::Values::NativeSeVolume, s.se_volume, opt->se_volume, 0, 10);
    sync_u8(BbSettings::Values::NativeVoiceVolume, s.voice_volume, opt->voice_volume, 0, 10);
    sync_bool(BbSettings::Values::NativeBloodExpression, s.blood_expression, opt->blood_expression, false);
    sync_bool(BbSettings::Values::NativeSubtitles, s.subtitles, opt->subtitles, false);
    sync_u8(BbSettings::Values::NativeGameBrightness, s.game_brightness, opt->brightness, 0, 10);
    sync_u8(BbSettings::Values::NativeCameraSpeed, s.camera_speed, opt->camera_speed, 0, 10);
    sync_u8(BbSettings::Values::NativePadVibration, s.pad_vibration, opt->pad_vibration, 0, 10);
    sync_bool(BbSettings::Values::NativeCameraInvertX, s.camera_invert_x, opt->camera_left_right, true);
    sync_bool(BbSettings::Values::NativeCameraInvertY, s.camera_invert_y, opt->camera_up_down, true);
    sync_bool(BbSettings::Values::NativeAutoTarget, s.auto_target, opt->auto_target, false);
    sync_bool(BbSettings::Values::NativeCameraAutoAdjust, s.camera_auto_adjust, opt->camera_auto_adjust, false);
    sync_u8(BbSettings::Values::NativeNetworkRegion, s.network_region, opt->network_region, 0, 1);
    sync_u8(BbSettings::Values::NativePlayerNameDisplay, s.player_name_display, opt->player_name_display, 0, 1);

    if (mask & BbSettings::Values::NativeNetworkPassword) {
        char16_t wanted_pw[9]{};
        Utf8ToPassword(BbSettings::GetNetworkPassword(), wanted_pw);
        if (std::memcmp(opt->password, wanted_pw, sizeof(wanted_pw)) != 0) {
            std::memcpy(opt->password, wanted_pw, sizeof(wanted_pw));
            modified_opt = true;
        }
    } else {
        BbSettings::SetNetworkPassword(PasswordToUtf8(opt->password));
    }

    if ((mask & BbSettings::Values::NativeGameTextLanguage) && s.game_text_language.load() >= 0) {
        const s32 wanted = std::clamp(s.game_text_language.load(), 0, BbSettings::GameLanguageCount - 1);
        if (opt->text_language != wanted) {
            opt->text_language = wanted;
            modified_opt = true;
        }
    } else if (!(mask & BbSettings::Values::NativeGameTextLanguage) &&
               opt->text_language >= 0 && opt->text_language < BbSettings::GameLanguageCount) {
        s.game_text_language.store(opt->text_language, std::memory_order_relaxed);
    }

    if ((mask & BbSettings::Values::NativeGameVoiceLanguage) && s.game_voice_language.load() >= 0) {
        const s32 wanted = std::clamp(s.game_voice_language.load(), 0, BbSettings::GameVoiceLanguageCount - 1);
        if (opt->voice_language != wanted) {
            opt->voice_language = wanted;
            modified_opt = true;
        }
    } else if (!(mask & BbSettings::Values::NativeGameVoiceLanguage) &&
               opt->voice_language >= 0 && opt->voice_language < BbSettings::GameVoiceLanguageCount) {
        s.game_voice_language.store(opt->voice_language, std::memory_order_relaxed);
    }

    const u32 all_bits = kAllNonLangNativeBits |
                         BbSettings::Values::NativeGameTextLanguage |
                         BbSettings::Values::NativeGameVoiceLanguage;
    const bool adopted_new_keys = (mask & all_bits) != all_bits;
    s.native_chosen_mask.fetch_or(all_bits, std::memory_order_relaxed);
    native_options_bootstrapped.store(true, std::memory_order_release);

    if (adopted_new_keys) {
        BbSettings::Save();
    }
    if (modified_opt) {
        QueueNativeOptionsDiskSync(*opt);
    }
}

/// Applies external Launcher edits from bbport.ini directly into live CSOptionsData in RAM.
void ApplySettingsToLiveNativeOptions(BbMenuEngine::CSOptionsData* opt) {
    if (!opt) return;
    const auto& s = BbSettings::Get();
    opt->music_volume = u8(std::clamp(s.bgm_volume.load(), 0, 10));
    opt->se_volume = u8(std::clamp(s.se_volume.load(), 0, 10));
    opt->voice_volume = u8(std::clamp(s.voice_volume.load(), 0, 10));
    opt->blood_expression = s.blood_expression.load() ? 1 : 0;
    opt->subtitles = s.subtitles.load() ? 1 : 0;
    opt->brightness = u8(std::clamp(s.game_brightness.load(), 0, 10));
    opt->camera_speed = u8(std::clamp(s.camera_speed.load(), 0, 10));
    opt->pad_vibration = u8(std::clamp(s.pad_vibration.load(), 0, 10));
    opt->camera_left_right = s.camera_invert_x.load() ? 0 : 1;
    opt->camera_up_down = s.camera_invert_y.load() ? 0 : 1;
    opt->auto_target = s.auto_target.load() ? 1 : 0;
    opt->camera_auto_adjust = s.camera_auto_adjust.load() ? 1 : 0;
    opt->network_region = u8(std::clamp(s.network_region.load(), 0, 1));
    opt->player_name_display = u8(std::clamp(s.player_name_display.load(), 0, 1));
    Utf8ToPassword(BbSettings::GetNetworkPassword(), opt->password);
    if (s.game_text_language.load() >= 0) {
        opt->text_language = std::clamp(s.game_text_language.load(), 0, BbSettings::GameLanguageCount - 1);
    }
    if (s.game_voice_language.load() >= 0) {
        opt->voice_language = std::clamp(s.game_voice_language.load(), 0, BbSettings::GameVoiceLanguageCount - 1);
    }
    if (in_controls_page.load(std::memory_order_relaxed) && active_game_settings) {
        active_game_settings[0x0] = opt->camera_speed;
        active_game_settings[0x1] = opt->pad_vibration;
        active_game_settings[0x9] = opt->camera_left_right;
        active_game_settings[0xa] = opt->camera_up_down;
        active_game_settings[0xb] = opt->auto_target;
        active_game_settings[0xc] = opt->camera_auto_adjust;
        RefreshControlsPage();
    }
    QueueNativeOptionsDiskSync(*opt);
}

/// Captures any in-game OptionPage commit (Environment, Brightness, Controls, Language, Network)
/// into BbSettings (bbport.ini -> Launcher) and queues immediate persistence to userdata0010.
void CaptureNativeOptionsFromGame(const BbMenuEngine::CSOptionsData& opt,
                                  const BbMenuEngine::CSOptionsData* prev_opt) {
    auto& s = BbSettings::Get();
    s.bgm_volume.store(std::clamp(int(opt.music_volume), 0, 10), std::memory_order_relaxed);
    s.se_volume.store(std::clamp(int(opt.se_volume), 0, 10), std::memory_order_relaxed);
    s.voice_volume.store(std::clamp(int(opt.voice_volume), 0, 10), std::memory_order_relaxed);
    s.blood_expression.store(opt.blood_expression != 0, std::memory_order_relaxed);
    s.subtitles.store(opt.subtitles != 0, std::memory_order_relaxed);
    s.game_brightness.store(std::clamp(int(opt.brightness), 0, 10), std::memory_order_relaxed);
    s.camera_speed.store(std::clamp(int(opt.camera_speed), 0, 10), std::memory_order_relaxed);
    s.pad_vibration.store(std::clamp(int(opt.pad_vibration), 0, 10), std::memory_order_relaxed);
    s.camera_invert_x.store(opt.camera_left_right == 0, std::memory_order_relaxed);
    s.camera_invert_y.store(opt.camera_up_down == 0, std::memory_order_relaxed);
    s.auto_target.store(opt.auto_target != 0, std::memory_order_relaxed);
    s.camera_auto_adjust.store(opt.camera_auto_adjust != 0, std::memory_order_relaxed);
    s.network_region.store(std::clamp(int(opt.network_region), 0, 1), std::memory_order_relaxed);
    s.player_name_display.store(std::clamp(int(opt.player_name_display), 0, 1), std::memory_order_relaxed);
    BbSettings::SetNetworkPassword(PasswordToUtf8(opt.password));

    if (!prev_opt || opt.text_language != prev_opt->text_language) {
        s.game_text_language.store(
            (opt.text_language >= 0 && opt.text_language < BbSettings::GameLanguageCount)
                ? int(opt.text_language)
                : -1,
            std::memory_order_relaxed);
    }
    if (!prev_opt || opt.voice_language != prev_opt->voice_language) {
        s.game_voice_language.store(
            (opt.voice_language >= 0 && opt.voice_language < BbSettings::GameVoiceLanguageCount)
                ? int(opt.voice_language)
                : -1,
            std::memory_order_relaxed);
    }

    s.native_chosen_mask.fetch_or(
        kAllNonLangNativeBits | BbSettings::Values::NativeGameTextLanguage |
            BbSettings::Values::NativeGameVoiceLanguage,
        std::memory_order_relaxed);
    BbSettings::Save();
    QueueNativeOptionsDiskSync(opt);
    RequestNativeSaveFlag();
}

void CommitActiveControlsNativeSettings() {
    if (!active_controls_page || !image_base) return;
    Game<void (PS4_SYSV_ABI*)(void*)>(BbMenuEngine::Vma::NotifyOptionPageChanged)(active_controls_page);
}

void ApplyControlsSlotChoiceIndex(u8* row, int slot, int choice_idx) {
    const int item_idx = std::clamp(controls_scroll + slot, 0, TotalControlsItems - 1);
    if (item_idx == ItemCameraFollow) {
        const int m = std::clamp(choice_idx, 0, BbSettings::CameraFollowCount - 1);
        BbSettings::Get().camera_follow_mode.store(m, std::memory_order_relaxed);
        BbSettings::Save();
        RefreshSlotTexts(slot);
        UpdateSlotRowDisplay(row, slot);
        return;
    }
    const int idx = std::clamp(choice_idx, 0, 1);
    bool native_changed = false;
    if (item_idx == ItemCameraX && active_game_settings) {
        active_game_settings[0x9] = idx == 0 ? 1 : 0; // 1 = Normal, 0 = Inverted (CSOptionsData +0x0a)
        native_changed = true;
    } else if (item_idx == ItemCameraY && active_game_settings) {
        active_game_settings[0xa] = idx == 0 ? 1 : 0; // 1 = Normal, 0 = Inverted (CSOptionsData +0x0b)
        native_changed = true;
    } else if (item_idx == ItemAutoTarget && active_game_settings) {
        active_game_settings[0xb] = idx == 1 ? 1 : 0; // 1 = On, 0 = Off
        native_changed = true;
    } else if (item_idx == ItemCameraAdjust && active_game_settings) {
        active_game_settings[0xc] = idx == 1 ? 1 : 0; // 1 = On, 0 = Off
        native_changed = true;
    }
    if (native_changed) {
        CommitActiveControlsNativeSettings();
    }
    RefreshSlotTexts(slot);
    UpdateSlotRowDisplay(row, slot);
}

void StepSlotSetting(u8* row, int slot, int delta, bool wrap) {
    const int item_idx = std::clamp(controls_scroll + slot, 0, TotalControlsItems - 1);
    bool is_slider = false;
    bool changed = false;
    if (item_idx == ItemMouseSpeed) {
        is_slider = true;
        const int cur = CurrentMouseSpeedStep();
        int step = cur + delta;
        step = wrap ? (step + MouseSensSteps) % MouseSensSteps : std::clamp(step, 0, MouseSensSteps - 1);
        if (step != cur) {
            SetMouseSpeedStep(step);
            changed = true;
        }
    } else if (item_idx == ItemVibration && active_game_settings) {
        is_slider = true;
        const int cur = std::clamp(int(active_game_settings[0x1]), 0, 10);
        int vib = cur + delta;
        vib = wrap ? (vib + 11) % 11 : std::clamp(vib, 0, 10);
        if (vib != cur) {
            active_game_settings[0x1] = u8(vib);
            changed = true;
        }
    } else if (item_idx == ItemCameraSens && active_game_settings) {
        is_slider = true;
        const int cur = std::clamp(int(active_game_settings[0x0]), 0, 10);
        int sens = cur + delta;
        sens = wrap ? (sens + 11) % 11 : std::clamp(sens, 0, 10);
        if (sens != cur) {
            active_game_settings[0x0] = u8(sens);
            changed = true;
        }
    } else if (item_idx == ItemCameraFollow) {
        auto& s = BbSettings::Get();
        const int cur = std::clamp(s.camera_follow_mode.load(std::memory_order_relaxed),
                                   0, BbSettings::CameraFollowCount - 1);
        int next = cur + delta;
        next = wrap ? (next + BbSettings::CameraFollowCount) % BbSettings::CameraFollowCount
                    : std::clamp(next, 0, BbSettings::CameraFollowCount - 1);
        if (next != cur) {
            s.camera_follow_mode.store(next, std::memory_order_relaxed);
            BbSettings::Save();
            changed = true;
        }
    } else if (item_idx == ItemCameraX && active_game_settings) {
        active_game_settings[0x9] = active_game_settings[0x9] ? 0 : 1;
        changed = true;
    } else if (item_idx == ItemCameraY && active_game_settings) {
        active_game_settings[0xa] = active_game_settings[0xa] ? 0 : 1;
        changed = true;
    } else if (item_idx == ItemAutoTarget && active_game_settings) {
        active_game_settings[0xb] = active_game_settings[0xb] ? 0 : 1;
        changed = true;
    } else if (item_idx == ItemCameraAdjust && active_game_settings) {
        active_game_settings[0xc] = active_game_settings[0xc] ? 0 : 1;
        changed = true;
    } else if (item_idx == ItemLoadingTipsTimer) {
        auto& s = BbSettings::Get();
        const bool next = !s.loading_tips_timer.load(std::memory_order_relaxed);
        s.loading_tips_timer.store(next, std::memory_order_relaxed);
        BbFastLoad::SyncLoadingTipsTimer(next);
        BbSettings::Save();
        changed = true;
    }
    if (changed) {
        if (item_idx == ItemVibration || item_idx == ItemCameraSens ||
            (item_idx >= ItemCameraX && item_idx <= ItemCameraAdjust)) {
            CommitActiveControlsNativeSettings();
        }
        if (is_slider) {
            PlayMenuSliderSound();
        } else {
            PlayMenuStepSound();
        }
    }
    RefreshSlotTexts(slot);
    UpdateSlotRowDisplay(row, slot);
}

// ---- Scrollable 9-item System -> Display Window ----
constexpr int GameAaEffectIdx = 4; // "effect_game_aa" in BbSettings::Effects
constexpr int FpsPresets[] = {0, 30, 60, 72, 75, 90, 100, 120, 144, 165, 170, 180, 200, 240, 280, 300, 360, 480};
constexpr int FpsPresetCount = int(sizeof(FpsPresets) / sizeof(FpsPresets[0]));
constexpr int AnisoValues[] = {1, 4, 8, 16};
constexpr int AnisoValueCount = 4;

enum NativeAaMode : int {
    AaOff = 0,
    AaFxaa = 1,
    AaTaa = 2,
    AaDlaa = 3,
    AaDlaaCnn = 4,
    AaSmaa = 5,
    AaSmaaT2x = 6,
};

enum DisplayUpscalerChoice : int {
    DispUpOff = 0,
    DispUpDlss4 = 1,   // Preset K = 11 (DLSS 4.0 Transformer)
    DispUpDlss45M = 2, // Preset M = 13 (DLSS 4.5 2nd-Gen Transformer)
    DispUpDlss45L = 3, // Preset L = 12 (DLSS 4.5 2nd-Gen Transformer Ultra)
    DispUpDlss3 = 4,   // Preset F = 6 (DLSS 3.7 CNN Fast / Low Load)
    DispUpFsr31 = 5,
    DispUpFsr4 = 6,
    DispUpFsr411 = 7,
};

int disp_aspect = BbSettings::AspectAuto;
int disp_output_idx = 0;
int disp_aa_mode = AaOff;
DisplayUpscalerChoice disp_upscaler_choice = DispUpOff;
int disp_upscale_preset = BbSettings::NativeAA;
std::atomic<bool> display_heavy_dirty{false};
std::atomic<long long> display_heavy_dirty_ms{0};

void MarkDisplayHeavyDirty() {
    display_heavy_dirty_ms.store(NowSteadyMs(), std::memory_order_relaxed);
    display_heavy_dirty.store(true, std::memory_order_release);
}

void SyncDisplayStateFromSettings() {
    if (display_heavy_dirty.load(std::memory_order_acquire)) {
        return;
    }
    const auto& s = BbSettings::Get();
    disp_aspect = std::clamp(s.aspect.load(), 0, BbSettings::AspectCount - 1);
    menu_outputs = BbSettings::Outputs(disp_aspect);
    disp_output_idx = MenuOutputIndex(
        disp_aspect, BbSettings::NearestOutput(disp_aspect, BbSettings::UnpackSize(s.output_res.load())));
    const int up = std::clamp(s.upscaler.load(), 0, BbSettings::UpscalerCount - 1);
    const int dlss_p = s.dlss_preset.load();
    const int preset = std::clamp(s.preset.load(), 0, BbSettings::PresetCount - 1);

    disp_aa_mode = std::clamp(s.aa_mode.load(), 0, BbSettings::AaCount - 1);
    disp_upscale_preset = preset;
    if (up == BbSettings::UpscalerOff || up == BbSettings::UpscalerTaa) {
        disp_upscaler_choice = DispUpOff;
    } else if (up == BbSettings::UpscalerDlss) {
        disp_upscaler_choice = dlss_p == 13                           ? DispUpDlss45M
                               : dlss_p == 12                         ? DispUpDlss45L
                               : (dlss_p == 6 || dlss_p == 5 || dlss_p == 0) ? DispUpDlss3
                                                                      : DispUpDlss4;
    } else if (up == BbSettings::UpscalerFsr4) {
        disp_upscaler_choice = DispUpFsr4;
    } else if (up == BbSettings::UpscalerFsr411) {
        disp_upscaler_choice = DispUpFsr411;
    } else {
        disp_upscaler_choice = DispUpFsr31;
    }
}

int CurrentSharpnessStep() {
    const auto& s = BbSettings::Get();
    return s.sharpen
               ? std::clamp(int(std::lround(s.sharpness / SharpnessStep)), 1, SharpnessSteps - 1)
               : 0;
}

void SetSharpnessStep(int step, bool save = true) {
    auto& s = BbSettings::Get();
    step = std::clamp(step, 0, SharpnessSteps - 1);
    s.sharpen = step != 0;
    if (step != 0) {
        s.sharpness = step * SharpnessStep;
    }
    values[Sharpness] = step;
    applied[Sharpness] = step;
    bytes[Sharpness] = u8(step);
    if (save) {
        BbSettings::Save();
    }
}

int GetDisplayItemSliderValue(int item_idx) {
    if (item_idx == DispItemSharpness) {
        return CurrentSharpnessStep();
    }
    return -1;
}

void SyncDisplaySlotSliderWidget(u8* row, int slot, bool is_focused) {
    const int item_idx = std::clamp(display_scroll + slot, 0, TotalDisplayItems - 1);
    const char16_t* combo_text =
        (item_idx == DispItemUiUpscale) ? TextOf(DisplaySlotVal + u32(slot)) : nullptr;
    SyncPageSlotSliderWidget(active_display_page, row, slot, GetDisplayItemSliderValue(item_idx), is_focused,
                             combo_text);
}

std::vector<int> BuildFpsValuesList(int* out_cur_idx) {
    const auto& s = BbSettings::Get();
    const bool is_auto = s.fps_limit_auto.load(std::memory_order_relaxed);
    std::vector<int> vals;
    vals.reserve(MaxChoicesPerSlot);
    vals.push_back(-1); // Auto (Monitor Hz)
    for (int p : FpsPresets) {
        vals.push_back(p);
    }
    const int mon_hz = BbSettings::MonitorRefreshHz();
    if (mon_hz >= 30 && std::find(vals.begin(), vals.end(), mon_hz) == vals.end() &&
        int(vals.size()) < MaxChoicesPerSlot) {
        vals.push_back(mon_hz);
    }
    const int cur = std::max(0, BbFrameRate::Limit());
    if (!is_auto && cur > 0 && std::find(vals.begin(), vals.end(), cur) == vals.end() &&
        int(vals.size()) < MaxChoicesPerSlot) {
        vals.push_back(cur);
    }
    // Keep -1 (Auto) and 0 (Uncapped) at the front, sort positive FPS targets ascending.
    if (vals.size() > 2) {
        std::sort(vals.begin() + 2, vals.end());
    }
    int cur_idx = 0;
    if (!is_auto) {
        for (int i = 1; i < int(vals.size()); ++i) {
            if (vals[i] == cur) {
                cur_idx = i;
                break;
            }
        }
    }
    if (out_cur_idx) *out_cur_idx = cur_idx;
    return vals;
}

std::vector<DisplayUpscalerChoice> GetAvailableUpscalerChoices() {
    const auto& s = BbSettings::Get();
    std::vector<DisplayUpscalerChoice> choices = {DispUpOff};
    if (s.dlss_supported.load()) {
        choices.push_back(DispUpDlss45M);
        choices.push_back(DispUpDlss45L);
        choices.push_back(DispUpDlss4);
        choices.push_back(DispUpDlss3);
    }
    choices.push_back(DispUpFsr31);
    if (!s.dlss_supported.load() || disp_upscaler_choice == DispUpFsr4 || disp_upscaler_choice == DispUpFsr411) {
        if (s.fsr4_supported.load()) choices.push_back(DispUpFsr4);
        if (s.fsr411_supported.load()) choices.push_back(DispUpFsr411);
    }
    return choices;
}

void UpscalerChoiceLabels(DisplayUpscalerChoice c, const char*& out_en, const char*& out_ru) {
    switch (c) {
    case DispUpOff:
        out_en = "Off (Native 100%)";
        out_ru = "Выкл (Натив 100%)";
        break;
    case DispUpDlss45M:
        out_en = "DLSS 4.5 (Preset M)";
        out_ru = "DLSS 4.5 (Preset M)";
        break;
    case DispUpDlss45L:
        out_en = "DLSS 4.5 (Preset L)";
        out_ru = "DLSS 4.5 (Preset L)";
        break;
    case DispUpDlss4:
        out_en = "DLSS 4 (Preset K)";
        out_ru = "DLSS 4 (Preset K)";
        break;
    case DispUpDlss3:
        out_en = "DLSS 3.7 CNN (Fast)";
        out_ru = "DLSS 3.7 CNN (Быстрый)";
        break;
    case DispUpFsr4:
        out_en = "FSR 4";
        out_ru = "FSR 4";
        break;
    case DispUpFsr411:
        out_en = "FSR 4.1.1";
        out_ru = "FSR 4.1.1";
        break;
    case DispUpFsr31:
    default:
        out_en = "FSR 3.1";
        out_ru = "FSR 3.1";
        break;
    }
}

void UpscalerModeLabels(int preset, const char*& out_en, const char*& out_ru) {
    switch (std::clamp(preset, int(BbSettings::NativeAA), int(BbSettings::UltraPerformance))) {
    case BbSettings::NativeAA:
        out_en = "Native 100% (DLAA)";
        out_ru = "Натив 100% (DLAA)";
        break;
    case BbSettings::Quality:
        out_en = "Quality (67% Scale)";
        out_ru = "Качество (67%)";
        break;
    case BbSettings::Balanced:
        out_en = "Balanced (58% Scale)";
        out_ru = "Баланс (58%)";
        break;
    case BbSettings::Performance:
        out_en = "Performance (50% Scale)";
        out_ru = "Производит. (50%)";
        break;
    case BbSettings::UltraPerformance:
    default:
        out_en = "Ultra Perf. (33% Scale)";
        out_ru = "Ультра-произв. (33%)";
        break;
    }
}

std::vector<int> BuildFrameGenModes(const BbSettings::Values& s) {
    std::vector<int> modes = {BbSettings::FrameGenOff};
    if (s.frame_generation_supported.load(std::memory_order_relaxed)) {
        modes.push_back(BbSettings::FrameGenFsr3);
        modes.push_back(BbSettings::FrameGenFsr3_3x);
        modes.push_back(BbSettings::FrameGenFsr3_4x);
    }
    if (s.dlssg_supported.load(std::memory_order_relaxed)) {
        const int max_multi = std::max(1, s.dlssg_max_multi_frame.load(std::memory_order_relaxed));
        modes.push_back(BbSettings::FrameGenDlss2x);
        if (max_multi >= 2) modes.push_back(BbSettings::FrameGenDlss3x);
        if (max_multi >= 3) modes.push_back(BbSettings::FrameGenDlss4x);
    }
    return modes;
}

void BuildDisplaySlotChoices(int slot, std::vector<u32>& out_choices, int& out_selected_idx) {
    out_choices.clear();
    out_selected_idx = 0;
    const int item_idx = std::clamp(display_scroll + slot, 0, TotalDisplayItems - 1);
    const u32 base_id = DisplaySlotChoices + u32(slot * MaxChoicesPerSlot);
    const auto& s = BbSettings::Get();
    auto add_choice = [&](const char* en, const char* ru) {
        if (int(out_choices.size()) >= MaxChoicesPerSlot) return;
        const u32 id = base_id + u32(out_choices.size());
        SetDynamic(id, en, ru);
        out_choices.push_back(id);
    };

    if (item_idx == DispItemAspect) {
        const std::string group = BbSettings::AspectKey(BbSettings::ResolvedAspect(BbSettings::AspectAuto));
        add_choice(("Auto (" + group + ")").c_str(), ("Авто (" + group + ")").c_str());
        for (int a = BbSettings::AspectAuto + 1; a < BbSettings::AspectCount; ++a) {
            const char* k = BbSettings::AspectKey(a);
            add_choice(k, k);
        }
        out_selected_idx = std::clamp(disp_aspect, 0, int(out_choices.size()) - 1);
        return;
    }

    if (item_idx == DispItemResolution) {
        const int n = std::clamp(menu_outputs.count, 1, MaxChoicesPerSlot);
        for (int i = 0; i < n; ++i) {
            const auto o = MenuOutput(disp_aspect, i);
            const std::string txt = fmt::format("{} x {}", o.width, o.height);
            add_choice(txt.c_str(), txt.c_str());
        }
        out_selected_idx = std::clamp(disp_output_idx, 0, int(out_choices.size()) - 1);
        return;
    }

    if (item_idx == DispItemWindowMode) {
        add_choice("Fullscreen", "Полный экран");
        add_choice("Borderless Windowed", "Окно без рамки");
        add_choice("Windowed", "В окне");
        out_selected_idx = std::clamp(s.window_mode.load(), 0, BbSettings::WindowModeCount - 1);
        return;
    }

    if (item_idx == DispItemFpsLimit) {
        int cur_idx = 0;
        const std::vector<int> fps_vals = BuildFpsValuesList(&cur_idx);
        const int mon_hz = BbSettings::MonitorRefreshHz();
        const int fg_mult = BbSettings::FrameGenMultiplier(BbSettings::EffectiveFrameGen());
        const bool fg_ui = fg_mult > 1;
        for (int fps : fps_vals) {
            if (fps < 0) {
                if (fg_ui && mon_hz >= 60) {
                    const int base_hz = std::max(30, mon_hz / fg_mult);
                    const std::string en = fmt::format("Auto ({} -> {} Hz)", base_hz, base_hz * fg_mult);
                    const std::string ru = fmt::format("Авто ({} -> {} Гц)", base_hz, base_hz * fg_mult);
                    add_choice(en.c_str(), ru.c_str());
                } else {
                    const std::string en = mon_hz >= 30 ? fmt::format("Auto ({} Hz)", mon_hz) : "Auto";
                    const std::string ru = mon_hz >= 30 ? fmt::format("Авто ({} Гц)", mon_hz) : "Авто";
                    add_choice(en.c_str(), ru.c_str());
                }
            } else if (fps == 0) {
                add_choice("Uncapped", "Без лимита");
            } else if (fg_ui) {
                const int base_fps = (mon_hz >= 60 && fps >= mon_hz) ? std::max(30, fps / fg_mult) : fps;
                const std::string en = fmt::format("{} -> {} FPS", base_fps, base_fps * fg_mult);
                add_choice(en.c_str(), en.c_str());
            } else {
                bool is_preset = false;
                for (int p : FpsPresets) {
                    if (p == fps) is_preset = true;
                }
                const std::string en = is_preset ? fmt::format("{} FPS", fps) : fmt::format("{} FPS (Custom)", fps);
                const std::string ru = is_preset ? fmt::format("{} FPS", fps) : fmt::format("{} FPS (Свой)", fps);
                add_choice(en.c_str(), ru.c_str());
            }
        }
        out_selected_idx = std::clamp(cur_idx, 0, int(out_choices.size()) - 1);
        return;
    }

    if (item_idx == DispItemVsync) {
        add_choice("Fast (Default)", "Быстрый (по умолч.)");
        add_choice("On", "Вкл");
        add_choice("Off", "Выкл");
        out_selected_idx = std::clamp(s.vsync_mode.load(), 0, BbSettings::VsyncModeCount - 1);
        return;
    }

    if (item_idx == DispItemLowLatency) {
        const int ll_backend = s.low_latency_backend.load(std::memory_order_relaxed);
        if (ll_backend == BbSettings::LowLatencyBackendNvReflex) {
            const int eff_ll = Vulkan::LowLatency::EffectiveMode(ll_backend, s.low_latency.load());
            if (BbSettings::IsDlssFrameGen(BbSettings::EffectiveFrameGen())) {
                add_choice("On", "Вкл");
                add_choice("On + Boost", "Вкл + Boost");
                out_selected_idx = (eff_ll == BbSettings::LowLatencyBoost) ? 1 : 0;
            } else {
                add_choice("Off", "Выкл");
                add_choice("On", "Вкл");
                add_choice("On + Boost", "Вкл + Boost");
                out_selected_idx = std::clamp(eff_ll, 0, BbSettings::LowLatencyCount - 1);
            }
        } else if (ll_backend == BbSettings::LowLatencyBackendAmdAntiLag) {
            add_choice("Off", "Выкл");
            add_choice("On", "Вкл");
            out_selected_idx = s.low_latency.load() != BbSettings::LowLatencyOff ? 1 : 0;
        } else {
            add_choice("Unavailable", "Недоступно");
            out_selected_idx = 0;
        }
        return;
    }

    if (item_idx == DispItemAntiAliasing) {
        std::vector<int> modes = {AaOff, AaFxaa, AaSmaa, AaSmaaT2x, AaTaa};
        add_choice("Off", "Выкл");
        add_choice("FXAA", "FXAA");
        add_choice("SMAA 1x", "SMAA 1x");
        add_choice("SMAA T2x", "SMAA T2x");
        add_choice("TAA", "TAA");
        if (s.dlss_supported.load()) {
            modes.push_back(AaDlaa);
            modes.push_back(AaDlaaCnn);
            add_choice("DLAA 4 (Transformer)", "DLAA 4 (Трансформер)");
            add_choice("DLAA 3 (CNN Fast)", "DLAA 3 (CNN Быстрый)");
        }
        int sel = 0;
        for (int i = 0; i < int(modes.size()); ++i) {
            if (modes[i] == disp_aa_mode) sel = i;
        }
        out_selected_idx = sel;
        return;
    }

    if (item_idx == DispItemUpscaler) {
        const auto choices = GetAvailableUpscalerChoices();
        int sel = 0;
        for (int i = 0; i < int(choices.size()); ++i) {
            const char *en = nullptr, *ru = nullptr;
            UpscalerChoiceLabels(choices[i], en, ru);
            add_choice(en, ru);
            if (choices[i] == disp_upscaler_choice) sel = i;
        }
        out_selected_idx = sel;
        return;
    }

    if (item_idx == DispItemUpscalerMode) {
        for (int p = int(BbSettings::NativeAA); p <= int(BbSettings::UltraPerformance); ++p) {
            const char *en = nullptr, *ru = nullptr;
            UpscalerModeLabels(p, en, ru);
            add_choice(en, ru);
        }
        out_selected_idx =
            std::clamp(disp_upscale_preset, int(BbSettings::NativeAA), int(BbSettings::UltraPerformance));
        return;
    }

    if (item_idx == DispItemFrameGeneration) {
        const auto modes = BuildFrameGenModes(s);
        if (modes.size() > 1) {
            const int eff = BbSettings::EffectiveFrameGen();
            int sel = 0;
            for (int i = 0; i < int(modes.size()); ++i) {
                add_choice(BbSettings::FrameGenLabel(modes[i], false),
                           BbSettings::FrameGenLabel(modes[i], true));
                if (modes[i] == eff) sel = i;
            }
            out_selected_idx = sel;
        } else {
            add_choice("Unavailable", "Недоступно");
            out_selected_idx = 0;
        }
        return;
    }

    if (item_idx == DispItemTextureQuality) {
        add_choice("Default", "Стандарт");
        add_choice("Medium", "Среднее");
        add_choice("Low", "Низкое");
        add_choice("Very Low", "Очень низкое");
        out_selected_idx = std::clamp(s.texture_quality.load(), 0, 3);
        return;
    }

    if (item_idx == DispItemShadowQuality) {
        add_choice("Off", "Выкл");
        add_choice("Low", "Низкое");
        add_choice("Default", "Стандарт");
        add_choice("High", "Высокое");
        add_choice("Ultra", "Ультра");
        out_selected_idx = std::clamp(s.shadow_quality.load(), 0, 4);
        return;
    }

    if (item_idx == DispItemAnisotropic) {
        add_choice("Off", "Выкл");
        add_choice("4x (Medium)", "4x (Средняя)");
        add_choice("8x (High)", "8x (Высокая)");
        add_choice("16x (Ultra)", "16x (Ультра)");
        const int aniso = s.anisotropic.load();
        out_selected_idx = aniso >= 16 ? 3 : aniso >= 8 ? 2 : aniso >= 4 ? 1 : 0;
        return;
    }

    if (item_idx == DispItemModelLod) {
        add_choice("Highest", "Максимальная");
        add_choice("Default", "Стандарт");
        add_choice("Lower", "Ниже");
        add_choice("Lowest", "Минимальная");
        const int lod = s.model_lod.load();
        out_selected_idx = lod <= -2 ? 0 : lod == 0 ? 1 : lod == 1 ? 2 : 3;
        return;
    }

    if (item_idx == DispItemUiUpscale) {
        add_choice("Off", "Выкл");
        add_choice("On (AI / SDF 2x-4x)", "Вкл (AI / SDF 2x-4x)");
        out_selected_idx = s.ui_upscale.load() ? 1 : 0;
        return;
    }

    if (item_idx == DispItemClothPhysics) {
        add_choice("Default", "Стандарт");
        add_choice("Native Smooth", "Плавная (любой FPS)");
        add_choice("Enhanced (XPBD)", "Улучшенная (XPBD)");
        const auto cur_mode = BbClothExp::GetMode();
        out_selected_idx = (cur_mode == BbClothExp::ClothExpMode::XpbdEnhanced) ? 2
                         : (cur_mode == BbClothExp::ClothExpMode::LocalFp64)    ? 1
                                                                                : 0;
    }
}

void RefreshDisplaySlotTexts(int slot) {
    const int item_idx = std::clamp(display_scroll + slot, 0, TotalDisplayItems - 1);
    const u32 name_id = DisplaySlotName + u32(slot);
    const u32 help_id = DisplaySlotHelp + u32(slot);
    const u32 val_id = DisplaySlotVal + u32(slot);
    const auto& s = BbSettings::Get();

    if (item_idx == DispItemAspect) {
        SetDynamic(name_id, "Aspect Ratio", "Соотношение сторон");
        SetDynamic(help_id,
                   "Of the final frame: wider sees more to the sides, the UI keeps to the screen edges",
                   "Готового кадра: шире — больше видно по бокам, интерфейс у краёв экрана");
        if (disp_aspect == BbSettings::AspectAuto) {
            const std::string group = BbSettings::AspectKey(BbSettings::ResolvedAspect(BbSettings::AspectAuto));
            SetDynamic(val_id, ("Auto (" + group + ")").c_str(), ("Авто (" + group + ")").c_str());
        } else {
            const char* k = BbSettings::AspectKey(disp_aspect);
            SetDynamic(val_id, k, k);
        }
        return;
    }

    if (item_idx == DispItemResolution) {
        SetDynamic(name_id, "Resolution", "Разрешение");
        SetDynamic(help_id, "Size of the final frame and UI (applies live without restart)",
                   "Размер готового кадра и интерфейса (применяется сразу без перезапуска)");
        const auto o = MenuOutput(disp_aspect, disp_output_idx);
        const std::string txt = fmt::format("{} x {}", o.width, o.height);
        SetDynamic(val_id, txt.c_str(), txt.c_str());
        return;
    }

    if (item_idx == DispItemWindowMode) {
        SetDynamic(name_id, "Window Mode", "Режим экрана");
        SetDynamic(help_id,
                   "Fullscreen, Borderless Windowed or Windowed mode (applies live, Alt+Enter toggles)",
                   "Полный экран, окно без рамки или оконный режим (применяется сразу, Alt+Enter)");
        const int wm = std::clamp(s.window_mode.load(), 0, BbSettings::WindowModeCount - 1);
        if (wm == BbSettings::WindowModeFullscreen) {
            SetDynamic(val_id, "Fullscreen", "Полный экран");
        } else if (wm == BbSettings::WindowModeBorderless) {
            SetDynamic(val_id, "Borderless Windowed", "Окно без рамки");
        } else {
            SetDynamic(val_id, "Windowed", "В окне");
        }
        return;
    }

    if (item_idx == DispItemFpsLimit) {
        SetDynamic(name_id, "FPS Limit", "Лимит FPS");
        SetDynamic(help_id,
                   "Maximum frame rate: Auto (matches monitor Hz), Uncapped, or fixed FPS (shows Base -> FG output when Frame Gen is On)",
                   "Лимит кадров: Авто (под герцовку монитора), без лимита или точный FPS (при генерации кадров показывает База -> Выход)");
        const int fps = BbFrameRate::Limit();
        const int mon_hz = BbSettings::MonitorRefreshHz();
        const int fg_mult = BbSettings::FrameGenMultiplier(BbSettings::EffectiveFrameGen());
        const bool fg_ui = fg_mult > 1;
        if (s.fps_limit_auto.load(std::memory_order_relaxed)) {
            if (fg_ui && mon_hz >= 60) {
                const int base_hz = std::max(30, mon_hz / fg_mult);
                const std::string en = fmt::format("Auto ({} -> {} Hz)", base_hz, base_hz * fg_mult);
                const std::string ru = fmt::format("Авто ({} -> {} Гц)", base_hz, base_hz * fg_mult);
                SetDynamic(val_id, en.c_str(), ru.c_str());
            } else {
                const std::string en = mon_hz >= 30 ? fmt::format("Auto ({} Hz)", mon_hz) : "Auto";
                const std::string ru = mon_hz >= 30 ? fmt::format("Авто ({} Гц)", mon_hz) : "Авто";
                SetDynamic(val_id, en.c_str(), ru.c_str());
            }
        } else if (fps <= 0) {
            SetDynamic(val_id, "Uncapped", "Без лимита");
        } else if (fg_ui) {
            const int base_fps = (mon_hz >= 60 && fps >= mon_hz) ? std::max(30, fps / fg_mult) : fps;
            const std::string txt = fmt::format("{} -> {} FPS", base_fps, base_fps * fg_mult);
            SetDynamic(val_id, txt.c_str(), txt.c_str());
        } else {
            bool is_preset = false;
            for (int p : FpsPresets) {
                if (p == fps) is_preset = true;
            }
            const std::string en = is_preset ? fmt::format("{} FPS", fps) : fmt::format("{} FPS (Custom)", fps);
            const std::string ru = is_preset ? fmt::format("{} FPS", fps) : fmt::format("{} FPS (Свой)", fps);
            SetDynamic(val_id, en.c_str(), ru.c_str());
        }
        return;
    }

    if (item_idx == DispItemVsync) {
        SetDynamic(name_id, "VSync", "Вертикальная синхронизация");
        SetDynamic(help_id,
                   "Fast (no tearing/drops), On (VSync), Off (lowest latency, best for G-Sync/FreeSync)",
                   "Быстрый (без разрывов и провалов), Вкл (VSync), Выкл (мин. задержка, для G-Sync/FreeSync)");
        const int vs = std::clamp(s.vsync_mode.load(), 0, BbSettings::VsyncModeCount - 1);
        if (vs == BbSettings::VsyncOn) {
            SetDynamic(val_id, "On", "Вкл");
        } else if (vs == BbSettings::VsyncOff) {
            SetDynamic(val_id, "Off", "Выкл");
        } else {
            SetDynamic(val_id, "Fast (Default)", "Быстрый (по умолч.)");
        }
        return;
    }

    if (item_idx == DispItemLowLatency) {
        const int ll_backend = s.low_latency_backend.load(std::memory_order_relaxed);
        if (ll_backend == BbSettings::LowLatencyBackendNvReflex) {
            SetDynamic(name_id, "NVIDIA Reflex", "NVIDIA Reflex");
            SetDynamic(help_id,
                       "Reduces input latency when GPU-bound (On) and keeps GPU boost clocks steady (On + Boost)",
                       "Снижает задержку ввода при высокой нагрузке GPU (Вкл) и удерживает частоты (Вкл + Boost)");
            const int ll = Vulkan::LowLatency::EffectiveMode(ll_backend, s.low_latency.load());
            if (ll == BbSettings::LowLatencyBoost) {
                SetDynamic(val_id, "On + Boost", "Вкл + Boost");
            } else if (ll == BbSettings::LowLatencyOn) {
                SetDynamic(val_id, "On", "Вкл");
            } else {
                SetDynamic(val_id, "Off", "Выкл");
            }
        } else if (ll_backend == BbSettings::LowLatencyBackendAmdAntiLag) {
            SetDynamic(name_id, "AMD Anti-Lag 2", "AMD Anti-Lag 2");
            SetDynamic(help_id,
                       "Reduces input latency on AMD GPUs when GPU-bound (applies live)",
                       "Снижает задержку ввода на видеокартах AMD при высокой нагрузке GPU (применяется сразу)");
            if (s.low_latency.load() != BbSettings::LowLatencyOff) {
                SetDynamic(val_id, "On", "Вкл");
            } else {
                SetDynamic(val_id, "Off", "Выкл");
            }
        } else {
            SetDynamic(name_id, "Low Latency", "Низкая задержка");
            SetDynamic(help_id,
                       "Requires NVIDIA Reflex (VK_NV_low_latency2) or AMD Anti-Lag 2 (VK_AMD_anti_lag)",
                       "Требуется поддержка NVIDIA Reflex (VK_NV_low_latency2) или AMD Anti-Lag 2 (VK_AMD_anti_lag)");
            SetDynamic(val_id, "Unavailable", "Недоступно");
        }
        return;
    }

    if (item_idx == DispItemAntiAliasing) {
        SetDynamic(name_id, "Anti-Aliasing", "Сглаживание");
        SetDynamic(help_id,
                   "Anti-Aliasing mode when Upscaler is Off (Off / FXAA / SMAA 1x / SMAA T2x / TAA / DLAA 4 / DLAA 3)",
                   "Режим сглаживания (Выкл / FXAA / SMAA 1x / SMAA T2x / TAA / DLAA 4 Трансформер / DLAA 3 CNN)");
        if (disp_aa_mode == AaOff) {
            SetDynamic(val_id, "Off", "Выкл");
        } else if (disp_aa_mode == AaFxaa) {
            SetDynamic(val_id, "FXAA", "FXAA");
        } else if (disp_aa_mode == AaSmaa) {
            SetDynamic(val_id, "SMAA 1x", "SMAA 1x");
        } else if (disp_aa_mode == AaSmaaT2x) {
            SetDynamic(val_id, "SMAA T2x", "SMAA T2x");
        } else if (disp_aa_mode == AaTaa) {
            SetDynamic(val_id, "TAA", "TAA");
        } else if (disp_aa_mode == AaDlaaCnn) {
            SetDynamic(val_id, "DLAA 3 (CNN Fast)", "DLAA 3 (CNN Быстрый)");
        } else {
            SetDynamic(val_id, "DLAA 4 (Transformer)", "DLAA 4 (Трансформер)");
        }
        return;
    }

    if (item_idx == DispItemUpscaler) {
        SetDynamic(name_id, "Upscaler", "Апскейлер");
        SetDynamic(help_id,
                   "Super-resolution upscaler (Off = Native 100%; DLSS 4.5 Preset M/L, DLSS 4 Preset K, FSR 3.1)",
                   "Апскейлер (Выкл = Натив 100%; DLSS 4.5 Preset M/L, DLSS 4 Preset K, FSR 3.1)");
        const char *en = nullptr, *ru = nullptr;
        UpscalerChoiceLabels(disp_upscaler_choice, en, ru);
        SetDynamic(val_id, en, ru);
        return;
    }

    if (item_idx == DispItemUpscalerMode) {
        SetDynamic(name_id, "Upscaler Mode", "Режим апскейлера");
        SetDynamic(help_id,
                   "3D render scale when Upscaler is active: Native 100% (DLAA), Quality (67%), Balanced (58%), Performance (50%), Ultra Perf. (33%)",
                   "Масштаб 3D-рендера: Натив 100% (DLAA), Качество (67%), Баланс (58%), Производ. (50%), Ультра (33%)");
        const char *p_en = nullptr, *p_ru = nullptr;
        UpscalerModeLabels(disp_upscale_preset, p_en, p_ru);
        SetDynamic(val_id, p_en, p_ru);
        return;
    }

    if (item_idx == DispItemSharpness) {
        SetDynamic(name_id, "Sharpness (RCAS)", "Резкость (RCAS)");
        SetDynamic(help_id, "Contrast-adaptive sharpening (RCAS): 0 off, 10 strongest (Left/Right)",
                   "Адаптивная резкость (RCAS): 0 выкл, 10 сильнее всего (Влево/Вправо)");
        const int step = CurrentSharpnessStep();
        const std::string txt = fmt::format("{}", step);
        SetDynamic(val_id, txt.c_str(), txt.c_str());
        return;
    }

    if (item_idx == DispItemFrameGeneration) {
        SetDynamic(name_id, "Frame Generation", "Генерация кадров");
        const auto modes = BuildFrameGenModes(s);
        if (modes.size() > 1) {
            SetDynamic(help_id,
                       "Generates intermediate frames (FSR 3.1 2x/3x/4x on any GPU; DLSS 2x/3x/4x on RTX 40/50+, auto-enables Reflex) with clean HUD separation.",
                       "Создаёт промежуточные кадры (FSR 3.1 2x/3x/4x на любой GPU; DLSS 2x/3x/4x на RTX 40/50+, авто-Reflex) с чистым HUD.");
            const int eff = BbSettings::EffectiveFrameGen();
            SetDynamic(val_id, BbSettings::FrameGenLabel(eff, false),
                       BbSettings::FrameGenLabel(eff, true));
        } else {
            SetDynamic(help_id,
                       "Requires a Vulkan GPU supporting FSR 3.1 compute or NVIDIA RTX 40+ for DLSS-G",
                       "Требуется видеокарта с поддержкой FSR 3.1 или NVIDIA RTX 40+ для DLSS-G");
            SetDynamic(val_id, "Unavailable", "Недоступно");
        }
        return;
    }

    if (item_idx == DispItemTextureQuality) {
        SetDynamic(name_id, "Texture Quality", "Качество текстур");
        SetDynamic(help_id,
                   "World texture resolution: Default (full), Medium, Low, or Very Low (resizes textures in VRAM immediately without restarting; saves up to ~1 GB VRAM)",
                   "Детализация текстур мира: Стандарт, Среднее, Низкое или Очень низкое (мгновенно перестраивает текстуры в VRAM без перезапуска; экономит до ~1 ГБ VRAM)");
        const int tq = std::clamp(s.texture_quality.load(), 0, 3);
        if (tq == 0) {
            SetDynamic(val_id, "Default", "Стандарт");
        } else if (tq == 1) {
            SetDynamic(val_id, "Medium", "Среднее");
        } else if (tq == 2) {
            SetDynamic(val_id, "Low", "Низкое");
        } else {
            SetDynamic(val_id, "Very Low", "Очень низкое");
        }
        return;
    }

    if (item_idx == DispItemShadowQuality) {
        SetDynamic(name_id, "Shadow Quality", "Качество теней");
        SetDynamic(help_id,
                   "Shadow map resolution & sharpness: Off (max FPS), Low (1K), Default (4K), High (6K Crisp), Ultra (8K Razor-Sharp) — applies live",
                   "Разрешение и чёткость теней: Выкл (макс. FPS), Низкое (1K), Стандарт (4K), Высокое (6K чёткие), Ультра (8K бритвенные) — применяется мгновенно");
        const int sq = std::clamp(s.shadow_quality.load(), 0, 4);
        if (sq == 0) {
            SetDynamic(val_id, "Off", "Выкл");
        } else if (sq == 1) {
            SetDynamic(val_id, "Low", "Низкое");
        } else if (sq == 2) {
            SetDynamic(val_id, "Default", "Стандарт");
        } else if (sq == 3) {
            SetDynamic(val_id, "High", "Высокое");
        } else {
            SetDynamic(val_id, "Ultra", "Ультра");
        }
        return;
    }

    if (item_idx == DispItemAnisotropic) {
        SetDynamic(name_id, "Anisotropic Filtering", "Фильтрация текстур");
        SetDynamic(help_id,
                   "Sharpens ground, cobblestones and wall textures at oblique angles (applies live)",
                   "Чёткость текстур брусчатки, земли и стен под острым углом (применяется мгновенно)");
        const int aniso = s.anisotropic.load();
        if (aniso >= 16) {
            SetDynamic(val_id, "16x (Ultra)", "16x (Ультра)");
        } else if (aniso >= 8) {
            SetDynamic(val_id, "8x (High)", "8x (Высокая)");
        } else if (aniso >= 4) {
            SetDynamic(val_id, "4x (Medium)", "4x (Средняя)");
        } else {
            SetDynamic(val_id, "Off", "Выкл");
        }
        return;
    }

    if (item_idx == DispItemModelLod) {
        SetDynamic(name_id, "Model Detail (LOD)", "Детализация моделей");
        SetDynamic(help_id, "Level of detail distance for 3D models and geometry (applies live)",
                   "Дальность максимальной детализации 3D-моделей (применяется мгновенно)");
        const int lod = s.model_lod.load();
        if (lod <= -2) {
            SetDynamic(val_id, "Highest", "Максимальная");
        } else if (lod == 0) {
            SetDynamic(val_id, "Default", "Стандарт");
        } else if (lod == 1) {
            SetDynamic(val_id, "Lower", "Ниже");
        } else {
            SetDynamic(val_id, "Lowest", "Минимальная");
        }
        return;
    }

    if (item_idx == DispItemUiUpscale) {
        SetDynamic(name_id, "UI & Icons Upscale", "Апскейл иконок и UI");
        SetDynamic(help_id,
                   "2x/4x GPU Super-Resolution (SDF + De-Block + Edge Reconstruction) for logos, icons & HUD (applies live)",
                   "2x/4x апскейл логотипа, иконок оружия/брони/предметов и HUD на видеокарте (применяется мгновенно)");
        if (s.ui_upscale.load()) {
            SetDynamic(val_id, "On (AI / SDF 2x-4x)", "Вкл (AI / SDF 2x-4x)");
        } else {
            SetDynamic(val_id, "Off", "Выкл");
        }
        return;
    }

    if (item_idx == DispItemClothPhysics) {
        SetDynamic(name_id, "Cloth Physics", "Физика ткани");
        SetDynamic(help_id,
                   "Cloth simulation engine: Default, Native Smooth (any-FPS), or Enhanced XPBD (softer natural drape, applies live)",
                   "Движок физики ткани: Стандарт, Плавная (любой FPS) или Улучшенная XPBD (мягче складки и драпировка, мгновенно)");
        const auto cur_mode = BbClothExp::GetMode();
        if (cur_mode == BbClothExp::ClothExpMode::XpbdEnhanced) {
            SetDynamic(val_id, "Enhanced (XPBD)", "Улучшенная (XPBD)");
        } else if (cur_mode == BbClothExp::ClothExpMode::LocalFp64) {
            SetDynamic(val_id, "Native Smooth", "Плавная (любой FPS)");
        } else {
            SetDynamic(val_id, "Default", "Стандарт");
        }
    }
}

void UpdateDisplaySlotRowDisplay(u8* row, int slot) {
    if (!row) return;
    const int item_idx = std::clamp(display_scroll + slot, 0, TotalDisplayItems - 1);
    if (item_idx == DispItemSharpness) {
        const std::vector<u32> single = {DisplaySlotVal + u32(slot), DisplaySlotVal + u32(slot)};
        PopulateSelectRowChoices(row, FirstDisplaySlot + slot, single, 0);
    } else {
        std::vector<u32> choice_ids;
        int selected_idx = 0;
        BuildDisplaySlotChoices(slot, choice_ids, selected_idx);
        PopulateSelectRowChoices(row, FirstDisplaySlot + slot, choice_ids, selected_idx);
    }
    const int sel = GetSelectedDisplayRow();
    SyncDisplaySlotSliderWidget(row, slot, slot == sel);
}

void RefreshDisplayPage() {
    display_scroll = std::clamp(display_scroll, 0, MaxDisplayScroll);
    for (int slot = 0; slot < DisplaySlotCount; ++slot) {
        RefreshDisplaySlotTexts(slot);
    }
    if (!ValidateActivePageRows(active_display_page, DisplaySlotCount, &values[FirstDisplaySlot],
                                display_slot_rows)) {
        return;
    }
    u8* const rows_begin = *reinterpret_cast<u8**>(active_display_page + 0xe58);
    for (int slot = 0; slot < DisplaySlotCount; ++slot) {
        u8* const entry = rows_begin + slot * 0x90;
        *reinterpret_cast<const char16_t**>(entry + 0x00) = TextOf(DisplaySlotName + u32(slot));
        *reinterpret_cast<const char16_t**>(entry + 0x40) = TextOf(DisplaySlotHelp + u32(slot));
    }
    Game<void (PS4_SYSV_ABI*)(void*)>(ListDraw)(active_display_page + 0xa70);
    Game<void (PS4_SYSV_ABI*)(void*)>(UpdatePageHelpBar)(active_display_page);
    for (int slot = 0; slot < DisplaySlotCount; ++slot) {
        UpdateDisplaySlotRowDisplay(display_slot_rows[slot], slot);
    }
    BbMenuEngine::SyncPortScrollBar(
        active_display_page + BbMenuEngine::OptionPageLayout::kRootMovieClip, image_base,
        display_scroll, DisplaySlotCount, TotalDisplayItems);
}

void InitDisplayPage(void* page) {
    ClearActiveCustomPages();
    active_display_page = static_cast<u8*>(page);
    display_scroll = 0;
    display_wrapped_from_top = false;
    LoadValues();
    for (int slot = 0; slot < DisplaySlotCount; ++slot) {
        RefreshDisplaySlotTexts(slot);
    }
    AddRows(page, pages[0].rows);
    if (ValidateActivePageRows(active_display_page, DisplaySlotCount, &values[FirstDisplaySlot],
                               display_slot_rows)) {
        for (int slot = 0; slot < DisplaySlotCount; ++slot) {
            UpdateDisplaySlotRowDisplay(display_slot_rows[slot], slot);
        }
    }
    BbMenuEngine::SyncPortScrollBar(
        active_display_page + BbMenuEngine::OptionPageLayout::kRootMovieClip, image_base,
        display_scroll, DisplaySlotCount, TotalDisplayItems);
}

void ApplyDisplaySlotChoiceIndex(u8* row, int slot, int choice_idx) {
    auto& s = BbSettings::Get();
    const int item_idx = std::clamp(display_scroll + slot, 0, TotalDisplayItems - 1);

    if (item_idx == DispItemAspect) {
        const int new_aspect = std::clamp(choice_idx, 0, BbSettings::AspectCount - 1);
        if (new_aspect != disp_aspect) {
            const BbSettings::OutputSize shown = MenuOutput(disp_aspect, disp_output_idx);
            disp_aspect = new_aspect;
            menu_outputs = BbSettings::Outputs(disp_aspect);
            disp_output_idx = MenuOutputIndex(disp_aspect, BbSettings::NearestOutput(disp_aspect, shown));
            MarkDisplayHeavyDirty();
        }
        RefreshDisplayPage();
        return;
    }

    if (item_idx == DispItemResolution) {
        const int n = std::max(1, menu_outputs.count);
        const int new_idx = std::clamp(choice_idx, 0, n - 1);
        if (new_idx != disp_output_idx) {
            disp_output_idx = new_idx;
            MarkDisplayHeavyDirty();
        }
        RefreshDisplayPage();
        return;
    }

    if (item_idx == DispItemWindowMode) {
        const int new_wm = std::clamp(choice_idx, 0, BbSettings::WindowModeCount - 1);
        if (new_wm != s.window_mode.load()) {
            s.window_mode = new_wm;
            s.window_mode_chosen = true;
            BbSettings::Save();
        }
        RefreshDisplaySlotTexts(slot);
        UpdateDisplaySlotRowDisplay(row, slot);
        return;
    }

    if (item_idx == DispItemFpsLimit) {
        const std::vector<int> fps_vals = BuildFpsValuesList(nullptr);
        if (!fps_vals.empty()) {
            const int chosen = fps_vals[std::clamp(choice_idx, 0, int(fps_vals.size()) - 1)];
            if (chosen < 0) {
                const int hz = BbSettings::MonitorRefreshHz();
                const int target = hz >= 30 ? hz : 0;
                const bool changed = !s.fps_limit_auto.load() || target != BbFrameRate::Limit();
                s.fps_limit_auto = true;
                s.fps_limit = target;
                BbFrameRate::SetLimit(target);
                if (changed) {
                    BbSettings::Save();
                }
            } else {
                const bool changed = s.fps_limit_auto.load() || chosen != BbFrameRate::Limit();
                s.fps_limit_auto = false;
                s.fps_limit = chosen;
                BbFrameRate::SetLimit(chosen);
                if (changed) {
                    BbSettings::Save();
                }
            }
        }
        RefreshDisplaySlotTexts(slot);
        UpdateDisplaySlotRowDisplay(row, slot);
        return;
    }

    if (item_idx == DispItemVsync) {
        const int new_vs = std::clamp(choice_idx, 0, BbSettings::VsyncModeCount - 1);
        if (new_vs != s.vsync_mode.load()) {
            s.vsync_mode = new_vs;
            s.vsync_mode_chosen = true;
            s.swapchain_generation.fetch_add(1, std::memory_order_relaxed);
            BbSettings::Save();
        }
        RefreshDisplaySlotTexts(slot);
        UpdateDisplaySlotRowDisplay(row, slot);
        return;
    }

    if (item_idx == DispItemLowLatency) {
        const int ll_backend = s.low_latency_backend.load(std::memory_order_relaxed);
        if (ll_backend == BbSettings::LowLatencyBackendNvReflex) {
            const int new_ll = BbSettings::IsDlssFrameGen(BbSettings::EffectiveFrameGen())
                                   ? (choice_idx != 0 ? BbSettings::LowLatencyBoost
                                                      : BbSettings::LowLatencyOn)
                                   : std::clamp(choice_idx, 0, BbSettings::LowLatencyCount - 1);
            if (new_ll != s.low_latency.load()) {
                s.low_latency = new_ll;
                BbSettings::Save();
                Vulkan::LowLatency::SyncModeIfChanged();
                BbSysInfo::LogSettingsIfChanged();
            }
        } else if (ll_backend == BbSettings::LowLatencyBackendAmdAntiLag) {
            const int new_ll = choice_idx != 0 ? BbSettings::LowLatencyOn : BbSettings::LowLatencyOff;
            if (new_ll != s.low_latency.load()) {
                s.low_latency = new_ll;
                BbSettings::Save();
                Vulkan::LowLatency::SyncModeIfChanged();
                BbSysInfo::LogSettingsIfChanged();
            }
        }
        RefreshDisplaySlotTexts(slot);
        UpdateDisplaySlotRowDisplay(row, slot);
        return;
    }

    if (item_idx == DispItemAntiAliasing) {
        std::vector<int> modes = {AaOff, AaFxaa, AaSmaa, AaSmaaT2x, AaTaa};
        if (s.dlss_supported.load()) {
            modes.push_back(AaDlaa);
            modes.push_back(AaDlaaCnn);
        }
        const int new_mode = modes[std::clamp(choice_idx, 0, int(modes.size()) - 1)];
        if (new_mode != disp_aa_mode) {
            disp_aa_mode = new_mode;
            MarkDisplayHeavyDirty();
        }
        RefreshDisplaySlotTexts(slot);
        UpdateDisplaySlotRowDisplay(row, slot);
        return;
    }

    if (item_idx == DispItemUpscaler) {
        const auto choices = GetAvailableUpscalerChoices();
        const auto new_up = choices[std::clamp(choice_idx, 0, int(choices.size()) - 1)];
        if (new_up != disp_upscaler_choice) {
            const bool was_off = (disp_upscaler_choice == DispUpOff);
            disp_upscaler_choice = new_up;
            if (new_up == DispUpOff) {
                disp_upscale_preset = BbSettings::NativeAA;
            } else if (was_off && disp_upscale_preset == BbSettings::NativeAA) {
                disp_upscale_preset = BbSettings::Quality;
            }
            MarkDisplayHeavyDirty();
        }
        RefreshDisplayPage();
        return;
    }

    if (item_idx == DispItemUpscalerMode) {
        const int new_preset = std::clamp(choice_idx, int(BbSettings::NativeAA), int(BbSettings::UltraPerformance));
        if (new_preset != disp_upscale_preset ||
            (new_preset != BbSettings::NativeAA && disp_upscaler_choice == DispUpOff)) {
            disp_upscale_preset = new_preset;
            if (new_preset != BbSettings::NativeAA && disp_upscaler_choice == DispUpOff) {
                disp_upscaler_choice = s.dlss_supported.load() ? DispUpDlss45M : DispUpFsr31;
            }
            MarkDisplayHeavyDirty();
        }
        RefreshDisplayPage();
        return;
    }

    if (item_idx == DispItemFrameGeneration) {
        const auto modes = BuildFrameGenModes(s);
        if (modes.size() > 1) {
            const int new_fg = modes[std::clamp(choice_idx, 0, int(modes.size()) - 1)];
            if (new_fg != s.frame_generation.load(std::memory_order_relaxed)) {
                s.frame_generation = new_fg;
                if (BbSettings::IsDlssFrameGen(new_fg) &&
                    s.low_latency_backend.load(std::memory_order_relaxed) ==
                        BbSettings::LowLatencyBackendNvReflex &&
                    s.low_latency.load(std::memory_order_relaxed) == BbSettings::LowLatencyOff) {
                    s.low_latency = BbSettings::LowLatencyOn;
                }
                BbGuestHooks::ApplyGameEffects();
                BbSettings::Save();
                Vulkan::LowLatency::SyncModeIfChanged();
                BbSysInfo::LogSettingsIfChanged();
            }
        }
        RefreshDisplayPage();
        return;
    }

    if (item_idx == DispItemTextureQuality) {
        const int new_tq = std::clamp(choice_idx, 0, 3);
        if (new_tq != s.texture_quality.load() || !s.texture_quality_chosen.load()) {
            s.texture_quality = new_tq;
            s.texture_quality_chosen = true;
            BbSettings::Save();
        }
        RefreshDisplaySlotTexts(slot);
        UpdateDisplaySlotRowDisplay(row, slot);
        return;
    }

    if (item_idx == DispItemShadowQuality) {
        const int old_sq = std::clamp(s.shadow_quality.load(), 0, 4);
        const int new_sq = std::clamp(choice_idx, 0, 4);
        if (new_sq != old_sq || !s.shadow_quality_chosen.load()) {
            s.shadow_quality = new_sq;
            s.shadow_quality_chosen = true;
            BbSettings::Save();
        }
        RefreshDisplaySlotTexts(slot);
        UpdateDisplaySlotRowDisplay(row, slot);
        return;
    }

    if (item_idx == DispItemAnisotropic) {
        const int new_aniso = AnisoValues[std::clamp(choice_idx, 0, AnisoValueCount - 1)];
        if (new_aniso != s.anisotropic.load()) {
            s.anisotropic = new_aniso;
            BbSettings::Save();
        }
        RefreshDisplaySlotTexts(slot);
        UpdateDisplaySlotRowDisplay(row, slot);
        return;
    }

    if (item_idx == DispItemModelLod) {
        const int idx = std::clamp(choice_idx, 0, 3);
        const int new_lod = LodValues[idx];
        if (new_lod != s.model_lod.load()) {
            s.model_lod = new_lod;
            values[ModelLod] = idx;
            applied[ModelLod] = idx;
            BbGuestHooks::ApplyGameEffects();
            BbSettings::Save();
        }
        RefreshDisplaySlotTexts(slot);
        UpdateDisplaySlotRowDisplay(row, slot);
        return;
    }

    if (item_idx == DispItemUiUpscale) {
        const bool new_ui_up = (choice_idx != 0);
        if (new_ui_up != s.ui_upscale.load()) {
            s.ui_upscale = new_ui_up;
            BbSettings::Save();
        }
        RefreshDisplaySlotTexts(slot);
        UpdateDisplaySlotRowDisplay(row, slot);
        return;
    }

    if (item_idx == DispItemClothPhysics) {
        const auto new_mode =
            (choice_idx == 2) ? BbClothExp::ClothExpMode::XpbdEnhanced
                              : (choice_idx == 1) ? BbClothExp::ClothExpMode::LocalFp64
                                                  : BbClothExp::ClothExpMode::Off;
        if (new_mode != BbClothExp::GetMode() ||
            s.cloth_exp_mode.load() != static_cast<int>(new_mode)) {
            BbClothExp::SetMode(new_mode);
            s.cloth_exp_mode = static_cast<int>(new_mode);
            s.cloth_exp_chosen = true;
            BbSettings::Save();
            BbSysInfo::LogSettingsIfChanged();
        }
        RefreshDisplaySlotTexts(slot);
        UpdateDisplaySlotRowDisplay(row, slot);
    }
}

void StepDisplaySlotSetting(u8* row, int slot, int delta, bool wrap) {
    const int item_idx = std::clamp(display_scroll + slot, 0, TotalDisplayItems - 1);
    if (item_idx == DispItemSharpness) {
        const int cur = CurrentSharpnessStep();
        int step = cur + delta;
        step = wrap ? (step + SharpnessSteps) % SharpnessSteps : std::clamp(step, 0, SharpnessSteps - 1);
        if (step != cur) {
            SetSharpnessStep(step);
            PlayMenuSliderSound();
        }
        RefreshDisplaySlotTexts(slot);
        UpdateDisplaySlotRowDisplay(row, slot);
        return;
    }
    std::vector<u32> choice_ids;
    int cur_idx = 0;
    BuildDisplaySlotChoices(slot, choice_ids, cur_idx);
    const int n = std::max(1, int(choice_ids.size()));
    const int next_idx = (cur_idx + delta + n) % n;
    if (next_idx != cur_idx) {
        PlayMenuStepSound();
        ApplyDisplaySlotChoiceIndex(row, slot, next_idx);
    }
}

struct EffectsItemMeta {
    const char* name_en;
    const char* name_ru;
    const char* help_en;
    const char* help_ru;
    bool default_on;
};

constexpr EffectsItemMeta EffectsItems[TotalEffectsItems] = {
    {"Chromatic Aberration", "Хром. аберрация",
     "Red-blue color fringing near screen edges (turning it off makes the image cleaner)",
     "Красно-синяя окантовка по краям кадра (отключение делает картинку заметно чище)", false},
    {"Depth of Field (DoF)", "Глубина резкости (DoF)",
     "Cinematic background blur for objects outside the camera focus plane",
     "Кинематографическое размытие дальнего фона и объектов вне фокуса камеры", true},
    {"Camera Motion Blur", "Размытие камеры",
     "FPS-normalized world blur when turning the camera (Off / Low / Default / High — 0 extra CPU draws)",
     "Размытие мира при повороте камеры под любой FPS (Выкл / Низкий / Стандарт / Высокий — 0 нагрузки CPU)", true},
    {"Object Motion Blur", "Размытие объектов",
     "Per-object velocity blur for characters, weapons, and cloth (Off / Low / Default / High — Off saves CPU)",
     "Пообъектное размытие персонажей, оружия и плащей (Выкл / Низкий / Стандарт / Высокий — Выкл экономит CPU)", true},
    {"Ambient Occlusion (SSAO)", "Затенение SSAO",
     "Soft contact shading in corners, cobblestones, and cloth folds (adds depth)",
     "Мягкое контактное затенение в углах, щелях брусчатки и складках одежды", true},
    {"Dynamic Shadows", "Динамические тени",
     "Real-time shadows cast by lanterns, torches, bonfires, and gunfire",
     "Тени в реальном времени от фонарей, факелов, костров и выстрелов", true},
    {"SSR Reflections", "Отражения SSR",
     "Hidden PS4 screen-space raymarched reflections on puddles, blood, and wet stone",
     "Скрытая на PS4 трассировка экранных отражений в лужах, крови и мокром камне", true},
    {"Bloom", "Свечение (Bloom)",
     "Soft glow halo around lanterns, fire, moonlight, and arcane effects",
     "Мягкий ореол свечения вокруг фонарей, огня, луны и магии", true},
    {"Enhanced GTAO / SSAO", "Улучшенный GTAO / SSAO",
     "Vulkan 1.3 Remaster: Interleaved Gradient Noise (no 64x64 grid) + fine micro-crevice AO",
     "Ремастер Vulkan 1.3: убирает сетку шума 64x64 от PS4 и добавляет микро-затенение складок", true},
    {"Contact Micro-Shadows", "Контактные микро-тени",
     "Vulkan 1.3 Remaster: 12-tap Golden-Angle Vogel soft PCF + 8-step screen-space contact shadows",
     "Ремастер Vulkan 1.3: мягкие тени (12-Tap Vogel PCF) и трассировка контактных микро-теней", true},
    {"High-Precision Ultra SSR", "Высокоточный Ultra SSR",
     "Vulkan 1.3 Remaster: 2x SSR ray step density (1.5 px/step) + 30 Hz dynamic environment cubemap",
     "Ремастер Vulkan 1.3: 2x плотность лучей SSR (1.5 пикс/шаг) и обновление карты окружения 30 Гц", true},
    {"True Black & Vibrance", "Глубокий чёрный и сочность",
     "Vulkan 1.3 Remaster: removes 2015 PS4 LCD grey black-floor haze after 3D LUT + subtle vibrance",
     "Ремастер Vulkan 1.3: убирает серую пелену старых LCD 2015 года после 3D LUT и добавляет сочность", true},
    {"Native HDR10 (PQ)", "Настоящий HDR10 (PQ)",
     "Vulkan 1.3 Remaster: 10-bit Rec.2020 ST.2084 PQ HDR output with highlight expansion (enable HDR in OS)",
     "Ремастер Vulkan 1.3: 10-битный вывод HDR10 (Rec.2020 PQ) с расширением яркости огня и луны (включи HDR в ОС)", false},
    {"PCSS Soft Shadows", "Мягкие тени PCSS",
     "Vulkan 1.3 Remaster: physical contact-hardening soft shadows for 4K sun/moon CSM and torch/lantern VSM",
     "Ремастер Vulkan 1.3: физически мягкие тени PCSS (чёткие у основания, мягкие вдали) для луны, факелов и костров", true},
    {"Screen-Space GI (SSGI)", "Отражённый свет (SSGI)",
     "Vulkan 1.3 Remaster: horizon-based indirect bounce lighting from fire, lanterns, torches, and sunset sky",
     "Ремастер Vulkan 1.3: живой отражённый свет (SSGI) от костров, факелов, фонарей и закатного неба на камни и одежду", true},
    {"Fire Heat Haze", "Тепловое марево огня",
     "Vulkan 1.3 Remaster: subtle convective hot-air refraction rising above bonfires and torches",
     "Ремастер Vulkan 1.3: мягкое конвекционное дрожание горячего воздуха над кострами и факелами", true},
};

constexpr int EffectsItemToEffectIdx(int item_idx) {
    switch (item_idx) {
    case 0: return 0; // effect_chromatic_aberration
    case 1: return 1; // effect_dof
    // 2 = Camera Motion Blur, 3 = Object Motion Blur (4-level Off/Low/Default/High)
    case 4: return 3; // effect_ssao
    case 5: return 5; // effect_dynamic_shadows (4 is effect_game_aa)
    case 6: return 6; // effect_ssr
    case 7: return 7; // effect_bloom
    default: return -1;
    }
}

bool GetEffectsItemActive(int item_idx) {
    const auto& s = BbSettings::Get();
    const int e = EffectsItemToEffectIdx(item_idx);
    if (e >= 0) return s.effects[e].load();
    switch (item_idx) {
    case 2: return s.motion_blur_camera.load() > 0;
    case 3: return s.motion_blur_object.load() > 0;
    case 8: return s.ssao_quality.load();
    case 9: return s.contact_shadows.load();
    case 10: return s.ssr_quality.load();
    case 11: return s.true_black_hdr.load();
    case 12: return s.hdr_output.load();
    case 13: return s.pcss_shadows.load();
    case 14: return s.ssgi_enabled.load();
    case 15: return s.fire_heat_haze.load();
    default: return false;
    }
}

void SyncMotionBlurLegacyFlag() {
    auto& s = BbSettings::Get();
    const bool any_on = (s.motion_blur_camera.load() > 0) || (s.motion_blur_object.load() > 0);
    s.effects[2] = any_on;
    values[FirstEffect + 2] = any_on ? 1 : 0;
    applied[FirstEffect + 2] = any_on ? 1 : 0;
    bytes[FirstEffect + 2] = u8(any_on ? 1 : 0);
}

void SetEffectsItemActive(int item_idx, bool active, bool commit = true) {
    auto& s = BbSettings::Get();
    const int e = EffectsItemToEffectIdx(item_idx);
    if (e >= 0) {
        s.effects[e] = active;
        values[FirstEffect + e] = active ? 1 : 0;
        applied[FirstEffect + e] = active ? 1 : 0;
        bytes[FirstEffect + e] = u8(active ? 1 : 0);
    } else {
        switch (item_idx) {
        case 2:
            s.motion_blur_camera = active ? 2 : 0; // Default when Reset to Defaults runs
            SyncMotionBlurLegacyFlag();
            break;
        case 3:
            s.motion_blur_object = active ? 2 : 0; // Default when Reset to Defaults runs
            SyncMotionBlurLegacyFlag();
            break;
        case 8: s.ssao_quality = active; break;
        case 9: s.contact_shadows = active; break;
        case 10: s.ssr_quality = active; break;
        case 11: s.true_black_hdr = active; break;
        case 12:
            if (s.hdr_output.load() != active) {
                s.hdr_output = active;
                s.swapchain_generation.fetch_add(1, std::memory_order_relaxed);
            }
            break;
        case 13: s.pcss_shadows = active; break;
        case 14: s.ssgi_enabled = active; break;
        case 15: s.fire_heat_haze = active; break;
        default: break;
        }
    }
    if (commit) {
        BbGuestHooks::ApplyGameEffects();
        BbSettings::Save();
    }
}

int GetMotionBlurItemChoiceIdx(int item_idx) {
    const auto& s = BbSettings::Get();
    if (item_idx == 2) return std::clamp(s.motion_blur_camera.load(), 0, 3);
    if (item_idx == 3) return std::clamp(s.motion_blur_object.load(), 0, 3);
    return 0;
}

void BuildEffectsSlotChoices(int slot, std::vector<u32>& out_choices, int& out_selected_idx) {
    out_choices.clear();
    const int item_idx = std::clamp(effects_scroll + slot, 0, TotalEffectsItems - 1);
    const u32 base_id = EffectsSlotChoices + u32(slot * 4);
    if (item_idx == 2 || item_idx == 3) {
        SetDynamic(base_id + 0, "Off", "Выкл");
        SetDynamic(base_id + 1, "Low", "Низкий");
        SetDynamic(base_id + 2, "Default", "Стандарт");
        SetDynamic(base_id + 3, "High", "Высокий");
        out_choices.push_back(base_id + 0);
        out_choices.push_back(base_id + 1);
        out_choices.push_back(base_id + 2);
        out_choices.push_back(base_id + 3);
        out_selected_idx = GetMotionBlurItemChoiceIdx(item_idx);
        return;
    }
    SetDynamic(base_id + 0, "Off", "Выкл");
    SetDynamic(base_id + 1, "On", "Вкл");
    out_choices.push_back(base_id + 0);
    out_choices.push_back(base_id + 1);
    out_selected_idx = GetEffectsItemActive(item_idx) ? 1 : 0;
}

void RefreshEffectsSlotTexts(int slot) {
    const int item_idx = std::clamp(effects_scroll + slot, 0, TotalEffectsItems - 1);
    const auto& meta = EffectsItems[item_idx];
    SetDynamic(EffectsSlotName + u32(slot), meta.name_en, meta.name_ru);
    SetDynamic(EffectsSlotHelp + u32(slot), meta.help_en, meta.help_ru);
    if (item_idx == 2 || item_idx == 3) {
        const int idx = GetMotionBlurItemChoiceIdx(item_idx);
        static constexpr const char* kEn[4] = {"Off", "Low", "Default", "High"};
        static constexpr const char* kRu[4] = {"Выкл", "Низкий", "Стандарт", "Высокий"};
        SetDynamic(EffectsSlotVal + u32(slot), kEn[idx], kRu[idx]);
        return;
    }
    const bool on = GetEffectsItemActive(item_idx);
    SetDynamic(EffectsSlotVal + u32(slot), on ? "On" : "Off", on ? "Вкл" : "Выкл");
}

void SyncEffectsSlotSliderWidget(u8* row, int slot, bool is_focused) {
    SyncPageSlotSliderWidget(active_effects_page, row, slot, -1, is_focused,
                             TextOf(EffectsSlotVal + u32(slot)));
}

void UpdateEffectsSlotRowDisplay(u8* row, int slot) {
    if (!row) return;
    std::vector<u32> choice_ids;
    int selected_idx = 0;
    BuildEffectsSlotChoices(slot, choice_ids, selected_idx);
    PopulateSelectRowChoices(row, FirstEffectsSlot + slot, choice_ids, selected_idx);
    const int sel = GetSelectedEffectsRow();
    SyncEffectsSlotSliderWidget(row, slot, slot == sel);
}

void RefreshEffectsPage() {
    effects_scroll = std::clamp(effects_scroll, 0, MaxEffectsScroll);
    for (int slot = 0; slot < EffectsSlotCount; ++slot) {
        RefreshEffectsSlotTexts(slot);
    }
    if (!ValidateActivePageRows(active_effects_page, EffectsSlotCount, &values[FirstEffectsSlot],
                                effects_slot_rows)) {
        return;
    }
    u8* const rows_begin = *reinterpret_cast<u8**>(active_effects_page + 0xe58);
    for (int slot = 0; slot < EffectsSlotCount; ++slot) {
        u8* const entry = rows_begin + slot * 0x90;
        *reinterpret_cast<const char16_t**>(entry + 0x00) = TextOf(EffectsSlotName + u32(slot));
        *reinterpret_cast<const char16_t**>(entry + 0x40) = TextOf(EffectsSlotHelp + u32(slot));
    }
    Game<void (PS4_SYSV_ABI*)(void*)>(ListDraw)(active_effects_page + 0xa70);
    Game<void (PS4_SYSV_ABI*)(void*)>(UpdatePageHelpBar)(active_effects_page);
    for (int slot = 0; slot < EffectsSlotCount; ++slot) {
        UpdateEffectsSlotRowDisplay(effects_slot_rows[slot], slot);
    }
    BbMenuEngine::SyncPortScrollBar(
        active_effects_page + BbMenuEngine::OptionPageLayout::kRootMovieClip, image_base,
        effects_scroll, EffectsSlotCount, TotalEffectsItems);
}

void InitEffectsPage(void* page) {
    ClearActiveCustomPages();
    active_effects_page = static_cast<u8*>(page);
    effects_scroll = 0;
    effects_wrapped_from_top = false;
    LoadValues();
    for (int slot = 0; slot < EffectsSlotCount; ++slot) {
        RefreshEffectsSlotTexts(slot);
    }
    AddRows(page, pages[1].rows);
    if (ValidateActivePageRows(active_effects_page, EffectsSlotCount, &values[FirstEffectsSlot],
                               effects_slot_rows)) {
        for (int slot = 0; slot < EffectsSlotCount; ++slot) {
            UpdateEffectsSlotRowDisplay(effects_slot_rows[slot], slot);
        }
    }
    BbMenuEngine::SyncPortScrollBar(
        active_effects_page + BbMenuEngine::OptionPageLayout::kRootMovieClip, image_base,
        effects_scroll, EffectsSlotCount, TotalEffectsItems);
}

void ApplyEffectsSlotChoiceIndex(u8* row, int slot, int choice_idx) {
    const int item_idx = std::clamp(effects_scroll + slot, 0, TotalEffectsItems - 1);
    if (item_idx == 2 || item_idx == 3) {
        const int clamped = std::clamp(choice_idx, 0, 3);
        auto& s = BbSettings::Get();
        auto& target = (item_idx == 2) ? s.motion_blur_camera : s.motion_blur_object;
        if (clamped != target.load()) {
            target = clamped;
            SyncMotionBlurLegacyFlag();
            BbGuestHooks::ApplyGameEffects();
            BbSettings::Save();
        }
        RefreshEffectsSlotTexts(slot);
        UpdateEffectsSlotRowDisplay(row, slot);
        return;
    }
    const bool new_active = (choice_idx != 0);
    if (new_active != GetEffectsItemActive(item_idx)) {
        SetEffectsItemActive(item_idx, new_active);
    }
    RefreshEffectsSlotTexts(slot);
    UpdateEffectsSlotRowDisplay(row, slot);
}

void StepEffectsSlotSetting(u8* row, int slot, int delta) {
    if (delta == 0) return;
    const int item_idx = std::clamp(effects_scroll + slot, 0, TotalEffectsItems - 1);
    if (item_idx == 2 || item_idx == 3) {
        const int cur = GetMotionBlurItemChoiceIdx(item_idx);
        const int next = (cur + (delta > 0 ? 1 : 3)) % 4;
        PlayMenuStepSound();
        ApplyEffectsSlotChoiceIndex(row, slot, next);
        return;
    }
    const bool cur = GetEffectsItemActive(item_idx);
    PlayMenuStepSound();
    ApplyEffectsSlotChoiceIndex(row, slot, cur ? 0 : 1);
}

void ShowCapturePrompt(u8* row, int slot, int k, const std::string& captured_text) {
    std::string primary, alt;
    SplitBinding(BbSettings::GetKeyBinding(k), primary, alt);
    std::string en, ru;
    if (!capturing_alt) {
        en = captured_text.empty() ? "[ Press Key... ]" : ("[ " + captured_text + " ... ]");
        ru = captured_text.empty() ? "[ Нажмите клавишу... ]" : ("[ " + captured_text + " ... ]");
    } else {
        const std::string a_en =
            captured_text.empty() ? "[ +Alt Key... ]" : ("[ " + captured_text + " ... ]");
        const std::string a_ru =
            captured_text.empty() ? "[ +2-я кл.... ]" : ("[ " + captured_text + " ... ]");
        en = BbSettings::FormatKeyBinding(primary, false) + " / " + a_en;
        ru = BbSettings::FormatKeyBinding(primary, true) + " / " + a_ru;
    }
    SetDynamic(ControlsSlotVal + u32(slot), en.c_str(), ru.c_str());
    UpdateSlotRowDisplay(row, slot);
}

void StepKeyCapture(u8* row, int slot, int k) {
    last_capture_step_ms.store(NowSteadyMs(), std::memory_order_relaxed);
    bool escape_down = false;
    std::vector<std::string> down = CollectDownTokens(&escape_down);

    if (capture_phase == CapturePhase::WaitInitialRelease) {
        captured_wheel.store(0, std::memory_order_relaxed);
        if (!escape_down && down.empty()) {
            capture_phase = CapturePhase::WaitPressAndRelease;
        }
        return;
    }
    if (capture_phase == CapturePhase::WaitEscapeRelease) {
        captured_wheel.store(0, std::memory_order_relaxed);
        if (!escape_down && down.empty()) {
            RefreshSlotTexts(slot);
            UpdateSlotRowDisplay(row, slot);
            capture_cooldown_ms.store(NowSteadyMs() + 300, std::memory_order_relaxed);
            capturing_action.store(-1, std::memory_order_relaxed);
        }
        return;
    }
    if (escape_down) {
        capture_phase = CapturePhase::WaitEscapeRelease;
        return;
    }
    const int wheel_dir = captured_wheel.exchange(0, std::memory_order_relaxed);
    if (wheel_dir != 0) {
        const std::string wheel_tok = wheel_dir > 0 ? "Wheel Up" : "Wheel Down";
        for (const std::string& tok : down) {
            if (IsModifierToken(tok) && captured_tokens.size() < 2 &&
                std::find(captured_tokens.begin(), captured_tokens.end(), tok) == captured_tokens.end()) {
                captured_tokens.push_back(tok);
            }
        }
        if (std::find(captured_tokens.begin(), captured_tokens.end(), wheel_tok) == captured_tokens.end()) {
            captured_tokens.push_back(wheel_tok);
        }
        down.clear(); // Commit wheel binding immediately
    }
    for (const std::string& tok : down) {
        if (captured_tokens.size() < 3 &&
            std::find(captured_tokens.begin(), captured_tokens.end(), tok) == captured_tokens.end()) {
            captured_tokens.push_back(tok);
        }
    }
    if (!down.empty() && !captured_tokens.empty()) {
        std::string preview;
        for (std::size_t i = 0; i < captured_tokens.size(); ++i) {
            if (i > 0) preview += " + ";
            preview += captured_tokens[i];
        }
        if (preview != last_capture_preview) {
            last_capture_preview = preview;
            ShowCapturePrompt(row, slot, k, preview);
        }
        return;
    }
    if (down.empty() && !captured_tokens.empty()) {
        // Order modifiers first so chords like Left Shift + Mouse Left are canonical.
        std::stable_partition(captured_tokens.begin(), captured_tokens.end(), IsModifierToken);
        std::string new_binding;
        for (std::size_t i = 0; i < captured_tokens.size(); ++i) {
            if (i > 0) new_binding += "+";
            new_binding += captured_tokens[i];
        }
        if (!capturing_alt && new_binding == "Right") {
            capturing_alt = true;
            captured_tokens.clear();
            last_capture_preview.clear();
            PlayMenuClickSound();
            ShowCapturePrompt(row, slot, k, {});
            return;
        }
        std::string primary, alt;
        SplitBinding(BbSettings::GetKeyBinding(k), primary, alt);
        if (capturing_alt) {
            if (new_binding == "Delete") {
                alt.clear();
            } else {
                alt = new_binding;
            }
            BbSettings::SetKeyBinding(k, JoinBinding(primary, alt));
        } else if (new_binding == "Delete") {
            BbSettings::SetKeyBinding(k, "");
        } else {
            BbSettings::SetKeyBinding(k, new_binding);
            if (k == BbSettings::KeyAdditionalDebug &&
                BbSettings::GetKeyBinding(BbSettings::KeyDebug) == new_binding) {
                BbSettings::SetKeyBinding(BbSettings::KeyDebug, "");
            } else if (k == BbSettings::KeyDebug &&
                       BbSettings::GetKeyBinding(BbSettings::KeyAdditionalDebug) == new_binding) {
                BbSettings::SetKeyBinding(BbSettings::KeyAdditionalDebug, "");
            }
        }
        BbSettings::Save();
        RefreshControlsPage();
        PlayMenuClickSound();
        capture_cooldown_ms.store(NowSteadyMs() + 300, std::memory_order_relaxed);
        capturing_action.store(-1, std::memory_order_relaxed);
    }
}

// ---- Hooks ----
using ControlsContentFn = void (PS4_SYSV_ABI*)(void*, u8*);
ControlsContentFn controls_content_original = nullptr;

void PS4_SYSV_ABI ControlsContentHook(void* page, u8* game_settings) {
    ClearActiveCustomPages();
    active_controls_page = static_cast<u8*>(page);
    in_controls_page.store(true, std::memory_order_relaxed);
    BbPrompts::SetControlsGuideActive(true);
    active_game_settings = game_settings;
    controls_scroll = 0;
    controls_wrapped_from_top = false;
    capture_cooldown_ms.store(NowSteadyMs() + 250, std::memory_order_relaxed);
    LoadValues();
    for (int slot = 0; slot < ControlsSlotCount; ++slot) {
        RefreshSlotTexts(slot);
    }
    AddRows(page, controls_rows);
    if (ValidateActivePageRows(active_controls_page, ControlsSlotCount, &values[FirstSlot], slot_rows)) {
        for (int slot = 0; slot < ControlsSlotCount; ++slot) {
            UpdateSlotRowDisplay(slot_rows[slot], slot);
        }
    }
    BbMenuEngine::SyncPortScrollBar(
        active_controls_page + BbMenuEngine::OptionPageLayout::kRootMovieClip, image_base,
        controls_scroll, ControlsSlotCount, TotalControlsItems);
}

using OptionsPageNavigateFn = void (PS4_SYSV_ABI*)(u8*, u8*, float);
OptionsPageNavigateFn options_page_navigate_original = nullptr;
std::atomic<long long> last_direct_wheel_ms{0};
const u8* scrollbar_drag_page = nullptr;

bool AnySlotPopupOpen(u8* const* rows, int slot_count) {
    for (int s = 0; s < slot_count; ++s) {
        if (rows[s] && rows[s][0x3c0] != 0) {
            return true;
        }
    }
    return false;
}

void HandleScrollablePageNavigate(u8* page, u8* input, float frame_time, int slot_count, int max_scroll,
                                  int& scroll, bool& wrapped_from_top, const int* first_val_ptr, u8** rows,
                                  int total_items, void (*refresh_fn)(), void (*sync_sliders_fn)(int)) {
    const bool popup_was_open = AnySlotPopupOpen(rows, slot_count);
    if (popup_was_open) {
        scrollbar_drag_page = nullptr;
    }

    if (!popup_was_open && max_scroll > 0) {
        last_direct_wheel_ms.store(NowSteadyMs(), std::memory_order_relaxed);
        if (const int wheel = BbMenuPointer::TakeWheelSteps(); wheel != 0) {
            BbMenuPointer::LockHoverAtCurrentPointer();
            int cur_sel = GetSelectedPageRow(page);
            bool changed = false;
            int rem = -wheel; // +down, -up
            if (rem < 0 && cur_sel == slot_count) {
                // Step off Defaults back onto the last visible setting row first.
                cur_sel = slot_count - 1;
                *reinterpret_cast<int*>(page + 0xae8) = -1;
                *reinterpret_cast<int*>(page + 0xae4) = cur_sel;
                BbMenuPointer::MoveListCursor(page + 0xa70, cur_sel);
                ++rem;
                changed = true;
            }
            if (rem != 0) {
                const int next_scroll = std::clamp(scroll + rem, 0, max_scroll);
                const int used = next_scroll - scroll;
                if (used != 0) {
                    scroll = next_scroll;
                    rem -= used;
                    changed = true;
                }
            }
            if (rem != 0) {
                const int next_sel = std::clamp(cur_sel + rem, 0, slot_count);
                if (next_sel != cur_sel) {
                    *reinterpret_cast<int*>(page + 0xae8) = -1;
                    *reinterpret_cast<int*>(page + 0xae4) = next_sel;
                    BbMenuPointer::MoveListCursor(page + 0xa70, next_sel);
                    changed = true;
                }
            }
            if (changed) {
                wrapped_from_top = false;
                refresh_fn();
                PlayMenuStepSound();
            }
        }

        if (!BbMenuPointer::LeftMouseHeld()) {
            scrollbar_drag_page = nullptr;
        } else if (BbMenuPointer::PointerActive()) {
            const float px = BbMenuPointer::PointerX();
            const float py = BbMenuPointer::PointerY();
            // Full-height PortScrollBarV in ControllSetting (Sprite 138 at stage_y = 206.7,
            // local y = -48.7, sy = 1.62 -> stage_y = 158.0, track span = 536 px).
            constexpr float sb_y = 158.0f;
            constexpr float track_top = sb_y + 8.0f;
            constexpr float track_h = 536.0f;
            constexpr float track_bot = track_top + track_h;
            const bool over_sb_x = (px >= 1396.0f && px <= 1446.0f);
            if (scrollbar_drag_page == page) {
                BbMenuPointer::ConsumePendingClick();
                const float ratio = std::clamp((py - track_top) / track_h, 0.0f, 1.0f);
                const int next_scroll = std::clamp(int(std::lround(ratio * float(max_scroll))), 0, max_scroll);
                if (next_scroll != scroll) {
                    scroll = next_scroll;
                    wrapped_from_top = false;
                    refresh_fn();
                    PlayMenuStepSound();
                }
                BbMenuEngine::SyncPortScrollBar(
                    page + BbMenuEngine::OptionPageLayout::kRootMovieClip, image_base,
                    scroll, slot_count, total_items);
                return;
            }
            if (over_sb_x && py >= sb_y - 24.0f && py <= track_bot + 32.0f &&
                BbMenuPointer::ConsumePendingClick()) {
                if (py < track_top) {
                    if (scroll > 0) {
                        --scroll;
                        wrapped_from_top = false;
                        refresh_fn();
                        PlayMenuStepSound();
                    }
                } else if (py > track_bot) {
                    if (scroll < max_scroll) {
                        ++scroll;
                        wrapped_from_top = false;
                        refresh_fn();
                        PlayMenuStepSound();
                    }
                } else {
                    scrollbar_drag_page = page;
                    const float ratio = std::clamp((py - track_top) / track_h, 0.0f, 1.0f);
                    const int next_scroll = std::clamp(int(std::lround(ratio * float(max_scroll))), 0, max_scroll);
                    if (next_scroll != scroll) {
                        scroll = next_scroll;
                        wrapped_from_top = false;
                        refresh_fn();
                        PlayMenuStepSound();
                    }
                }
                BbMenuEngine::SyncPortScrollBar(
                    page + BbMenuEngine::OptionPageLayout::kRootMovieClip, image_base,
                    scroll, slot_count, total_items);
                return;
            }
        }
    }

    const int pre_sel = GetSelectedPageRow(page);
    options_page_navigate_original(page, input, frame_time);
    if (!ValidateActivePageRows(page, slot_count, first_val_ptr, rows)) {
        return;
    }
    BbMenuEngine::SyncPortScrollBar(
        page + BbMenuEngine::OptionPageLayout::kRootMovieClip, image_base,
        scroll, slot_count, total_items);
    if (popup_was_open || AnySlotPopupOpen(rows, slot_count)) {
        return;
    }
    const int hover_sel = BbMenuPointer::PostHoverCursor(page + 0xa70, pre_sel);
    int new_sel = GetSelectedPageRow(page);
    if (pre_sel == slot_count && hover_sel >= 0 && hover_sel < slot_count && new_sel == slot_count) {
        // Mouse pointer or wheel moved from Defaults onto one of the list rows: activate that row.
        *reinterpret_cast<int*>(page + 0xae8) = -1;
        *reinterpret_cast<int*>(page + 0xae4) = hover_sel;
        new_sel = hover_sel;
        wrapped_from_top = false;
        Game<void (PS4_SYSV_ABI*)(void*)>(UpdatePageHelpBar)(page);
        sync_sliders_fn(new_sel);
        return;
    }
    if (new_sel != hover_sel) {
        if (pre_sel == slot_count - 1 && hover_sel == slot_count - 1 && new_sel == slot_count && scroll < max_scroll) {
            ++scroll;
            wrapped_from_top = false;
            *reinterpret_cast<int*>(page + 0xae8) = -1;
            Game<int (PS4_SYSV_ABI*)(void*, u32)>(ListSelect)(page + 0xa70, u32(slot_count - 1));
            Game<void (PS4_SYSV_ABI*)(void*)>(ListShow)(page + 0xa70);
            refresh_fn();
            return;
        }
        if (pre_sel == 0 && hover_sel == 0 && new_sel == slot_count) {
            if (scroll > 0) {
                --scroll;
                wrapped_from_top = false;
                *reinterpret_cast<int*>(page + 0xae8) = -1;
                Game<int (PS4_SYSV_ABI*)(void*, u32)>(ListSelect)(page + 0xa70, 0);
                Game<void (PS4_SYSV_ABI*)(void*)>(ListShow)(page + 0xa70);
                refresh_fn();
                return;
            }
            wrapped_from_top = true;
        } else if (hover_sel == slot_count && new_sel == slot_count - 1 &&
                   wrapped_from_top && scroll != max_scroll) {
            scroll = max_scroll;
            wrapped_from_top = false;
            refresh_fn();
            return;
        } else if (hover_sel == slot_count && new_sel == 0 && scroll != 0) {
            scroll = 0;
            wrapped_from_top = false;
            refresh_fn();
            return;
        } else {
            wrapped_from_top = false;
        }
    } else if (new_sel != pre_sel) {
        wrapped_from_top = false;
    }
    sync_sliders_fn(new_sel);
}

void SyncAllControlsSliders(int new_sel) {
    if (!ValidateActivePageRows(active_controls_page, ControlsSlotCount, &values[FirstSlot], slot_rows)) {
        return;
    }
    for (int slot = 0; slot < ControlsSlotCount; ++slot) {
        SyncSlotSliderWidget(slot_rows[slot], slot, slot == new_sel);
    }
}

void SyncAllDisplaySliders(int new_sel) {
    if (!ValidateActivePageRows(active_display_page, DisplaySlotCount, &values[FirstDisplaySlot],
                                display_slot_rows)) {
        return;
    }
    for (int slot = 0; slot < DisplaySlotCount; ++slot) {
        SyncDisplaySlotSliderWidget(display_slot_rows[slot], slot, slot == new_sel);
    }
}

void SyncAllEffectsSliders(int new_sel) {
    if (!ValidateActivePageRows(active_effects_page, EffectsSlotCount, &values[FirstEffectsSlot],
                                effects_slot_rows)) {
        return;
    }
    for (int slot = 0; slot < EffectsSlotCount; ++slot) {
        SyncEffectsSlotSliderWidget(effects_slot_rows[slot], slot, slot == new_sel);
    }
}

void ResetDisplayPageToDefaults() {
    auto& s = BbSettings::Get();
    disp_aspect = BbSettings::AspectAuto;
    menu_outputs = BbSettings::Outputs(disp_aspect);
    disp_output_idx =
        MenuOutputIndex(disp_aspect, BbSettings::NearestOutput(disp_aspect, BbSettings::Monitor()));
    s.output_res = BbSettings::PackSize(MenuOutput(disp_aspect, disp_output_idx));
    s.output_res_auto = true;
    disp_upscaler_choice = DispUpOff;
    disp_upscale_preset = BbSettings::NativeAA;
    disp_aa_mode = s.dlss_supported.load() ? AaDlaaCnn : AaSmaaT2x;
    MarkDisplayHeavyDirty();
    CommitDebouncedDisplaySettings(s, true);
    s.output_res_auto = true;
    s.window_mode = BbSettings::WindowModeBorderless;
    s.window_mode_chosen = true;
    s.fps_limit_auto = true;
    const int mon_hz = BbSettings::MonitorRefreshHz();
    const int target_fps = (mon_hz >= 30) ? mon_hz : 0;
    s.fps_limit = target_fps;
    BbFrameRate::SetLimit(target_fps);
    SetSharpnessStep(4, false);
    s.texture_quality = 0;
    s.texture_quality_chosen = false;
    s.shadow_quality = 2;
    s.shadow_quality_chosen = false;
    s.anisotropic = 16;
    s.model_lod = -2;
    values[ModelLod] = 0;
    applied[ModelLod] = 0;
    BbGuestHooks::ApplyGameEffects();
    s.ui_upscale = true;
    BbClothExp::SetMode(BbClothExp::ClothExpMode::Off);
    s.cloth_exp_mode = static_cast<int>(BbClothExp::ClothExpMode::Off);
    s.cloth_exp_chosen = true;
    s.low_latency = BbSettings::LowLatencyOn;
    s.frame_generation = BbSettings::FrameGenOff;
    if (s.vsync_mode.load() != BbSettings::VsyncFast) {
        s.vsync_mode = BbSettings::VsyncFast;
        s.vsync_mode_chosen = true;
        s.swapchain_generation.fetch_add(1, std::memory_order_relaxed);
    }
    BbSettings::Save();
    BbSysInfo::LogSettingsIfChanged();
    RefreshDisplayPage();
}

void ResetEffectsPageToDefaults() {
    for (int i = 0; i < TotalEffectsItems; ++i) {
        SetEffectsItemActive(i, EffectsItems[i].default_on, false);
    }
    BbGuestHooks::ApplyGameEffects();
    BbSettings::Save();
    BbSysInfo::LogSettingsIfChanged();
    RefreshEffectsPage();
}

void ResetControlsPageToDefaults() {
    for (int k = 0; k < BbSettings::KeyActionCount; ++k) {
        BbSettings::SetKeyBinding(k, BbSettings::GetKeyActionDef(k).default_val);
    }
    BbSettings::Get().camera_follow_mode.store(BbSettings::CameraFollowBalanced,
                                               std::memory_order_relaxed);
    BbSettings::Get().loading_tips_timer.store(false, std::memory_order_relaxed);
    BbFastLoad::SyncLoadingTipsTimer(false);
    SetMouseSpeedStep(5); // also saves bbport.ini
    if (active_game_settings) {
        active_game_settings[0x0] = 5;  // Camera Sensitivity
        active_game_settings[0x1] = 10; // Controller Vibration
        active_game_settings[0x9] = 1;  // Camera X-Axis (1 = Normal)
        active_game_settings[0xa] = 1;  // Camera Y-Axis (1 = Normal)
        active_game_settings[0xb] = 0;  // Auto Target Switch (0 = Off, matching CSOptionsData::InitDefaults)
        active_game_settings[0xc] = 1;  // Camera Auto-Adjust (1 = On)
        CommitActiveControlsNativeSettings();
    }
    RefreshControlsPage();
}

using OptionsPageResetDefaultsFn = void (PS4_SYSV_ABI*)(u8*);
OptionsPageResetDefaultsFn options_page_reset_defaults_original = nullptr;

void PS4_SYSV_ABI OptionsPageResetDefaultsHook(u8* page) {
    if (page == active_display_page &&
        ValidateActivePageRows(page, DisplaySlotCount, &values[FirstDisplaySlot], display_slot_rows)) {
        ResetDisplayPageToDefaults();
        return;
    }
    if (page == active_effects_page &&
        ValidateActivePageRows(page, EffectsSlotCount, &values[FirstEffectsSlot], effects_slot_rows)) {
        ResetEffectsPageToDefaults();
        return;
    }
    if (page == active_controls_page &&
        ValidateActivePageRows(page, ControlsSlotCount, &values[FirstSlot], slot_rows)) {
        ResetControlsPageToDefaults();
        return;
    }
    options_page_reset_defaults_original(page);
}

void PS4_SYSV_ABI OptionsPageNavigateHook(u8* page, u8* input, float frame_time) {
    BbMenuPointer::NoteMenuActive();
    static const u8* last_option_page = nullptr;
    if (page != last_option_page) {
        last_option_page = page;
        BbMenuPointer::TakeWheelSteps();
    }
    if (page == active_controls_page) {
        if (!ValidateActivePageRows(page, ControlsSlotCount, &values[FirstSlot], slot_rows)) {
            active_controls_page = nullptr;
            in_controls_page.store(false, std::memory_order_relaxed);
            BbPrompts::SetControlsGuideActive(false);
            for (int i = 0; i < ControlsSlotCount; ++i) slot_rows[i] = nullptr;
        } else {
            in_controls_page.store(true, std::memory_order_relaxed);
            const int sel = GetSelectedControlsRow();
            const int item_idx = std::clamp(controls_scroll + std::max(0, sel), 0, TotalControlsItems - 1);
            BbPrompts::SetControlsGuideActive(sel >= 0 && sel < ControlsSlotCount && ItemKeyAction(item_idx) >= 0);
            if (capturing_action.load(std::memory_order_relaxed) >= 0 || InCaptureCooldown()) {
                return;
            }
            HandleScrollablePageNavigate(page, input, frame_time, ControlsSlotCount, MaxControlsScroll,
                                         controls_scroll, controls_wrapped_from_top, &values[FirstSlot],
                                         slot_rows, TotalControlsItems, &RefreshControlsPage, &SyncAllControlsSliders);
            if (ValidateActivePageRows(page, ControlsSlotCount, &values[FirstSlot], slot_rows)) {
                const int new_sel = GetSelectedControlsRow();
                const int new_idx = std::clamp(controls_scroll + std::max(0, new_sel), 0, TotalControlsItems - 1);
                BbPrompts::SetControlsGuideActive(new_sel >= 0 && new_sel < ControlsSlotCount &&
                                                  ItemKeyAction(new_idx) >= 0);
                Game<void (PS4_SYSV_ABI*)(void*)>(UpdatePageButtonGuide)(page);
            }
            return;
        }
    }
    in_controls_page.store(false, std::memory_order_relaxed);
    BbPrompts::SetControlsGuideActive(false);
    if (page == active_display_page) {
        if (!ValidateActivePageRows(page, DisplaySlotCount, &values[FirstDisplaySlot], display_slot_rows)) {
            active_display_page = nullptr;
            for (int i = 0; i < DisplaySlotCount; ++i) display_slot_rows[i] = nullptr;
        } else {
            HandleScrollablePageNavigate(page, input, frame_time, DisplaySlotCount, MaxDisplayScroll,
                                         display_scroll, display_wrapped_from_top, &values[FirstDisplaySlot],
                                         display_slot_rows, TotalDisplayItems, &RefreshDisplayPage, &SyncAllDisplaySliders);
            return;
        }
    }
    if (page == active_effects_page) {
        if (!ValidateActivePageRows(page, EffectsSlotCount, &values[FirstEffectsSlot], effects_slot_rows)) {
            active_effects_page = nullptr;
            for (int i = 0; i < EffectsSlotCount; ++i) effects_slot_rows[i] = nullptr;
        } else {
            HandleScrollablePageNavigate(page, input, frame_time, EffectsSlotCount, MaxEffectsScroll,
                                         effects_scroll, effects_wrapped_from_top, &values[FirstEffectsSlot],
                                         effects_slot_rows, TotalEffectsItems, &RefreshEffectsPage, &SyncAllEffectsSliders);
            return;
        }
    }
    scrollbar_drag_page = nullptr;
    options_page_navigate_original(page, input, frame_time);
}

using RowInputFn = u64 (PS4_SYSV_ABI*)(u8*, u64, u64, u64, u64, u64, double);
RowInputFn row_input_original = nullptr;

int HorizontalMenuStep(u8 in) {
    if (in == 0) return 0;
    if (Game<bool (PS4_SYSV_ABI*)(const u8*)>(IsLeftPressed)(&in)) return -1;
    if (Game<bool (PS4_SYSV_ABI*)(const u8*)>(IsRightPressed)(&in)) return 1;
    const std::int32_t step = Game<std::int32_t (PS4_SYSV_ABI*)(const u8*)>(GetHorizontalStep)(&in);
    return step < 0 ? -1 : (step > 0 ? 1 : 0);
}

u64 PS4_SYSV_ABI RowInputHook(u8* row, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, double frame_time) {
    int* const val_ptr = *reinterpret_cast<int**>(row + RowValue);
    // The game asks the open page's rows: the resolution row is alive while it does.
    if (val_ptr == &values[OutputRes]) {
        RefreshOutputRow(row);
    }
    if (val_ptr >= &values[FirstDisplaySlot] &&
        val_ptr < &values[int(FirstDisplaySlot) + DisplaySlotCount]) {
        const int slot = int(val_ptr - &values[FirstDisplaySlot]);
        display_slot_rows[slot] = row;
        if (values[FirstDisplaySlot + slot] == DefaultResetSentinel) {
            ResetDisplayPageToDefaults();
            return 0;
        }
        const int item_idx = std::clamp(display_scroll + slot, 0, TotalDisplayItems - 1);
        const u8 in = u8(a1);
        const bool is_focused = (slot == GetSelectedDisplayRow());
        if (item_idx == DispItemSharpness) {
            // Sharpness slider row: Left/Right step the slider.
            SyncDisplaySlotSliderWidget(row, slot, is_focused);
            if (is_focused && in != 0) {
                const int step = HorizontalMenuStep(in);
                if (step != 0) {
                    StepDisplaySlotSetting(row, slot, step < 0 ? -1 : 1, false);
                }
            }
            return 0;
        }
        if (item_idx == DispItemUiUpscale) {
            // 2-choice toggle row (UI & Icons Upscale): uses <| Off / On |> ComboBox arrows.
            SyncDisplaySlotSliderWidget(row, slot, is_focused);
            if (is_focused && in != 0) {
                const int step = HorizontalMenuStep(in);
                const bool confirm =
                    (step == 0) && Game<bool (PS4_SYSV_ABI*)(const u8*)>(IsConfirmPressed)(&in);
                if (step != 0 || confirm) {
                    StepDisplaySlotSetting(row, slot, step != 0 ? step : 1, true);
                }
            }
            return 0;
        }
        // Multi-choice Display row (3+ options): supports both native pop-up dropdown (Confirm) and quick Left/Right step.
        if (row[0x3c0] != 0) {
            const u64 res = row_input_original(row, a1, a2, a3, a4, a5, frame_time);
            if (row[0x3c0] == 0) {
                ApplyDisplaySlotChoiceIndex(row, slot, values[FirstDisplaySlot + slot]);
            }
            return res;
        }
        if (is_focused && in != 0) {
            const int step = HorizontalMenuStep(in);
            if (step != 0) {
                StepDisplaySlotSetting(row, slot, step, false);
                return 0;
            }
        }
        const int before_val = values[FirstDisplaySlot + slot];
        const u64 res = row_input_original(row, a1, a2, a3, a4, a5, frame_time);
        if (row[0x3c0] == 0 && values[FirstDisplaySlot + slot] != before_val) {
            ApplyDisplaySlotChoiceIndex(row, slot, values[FirstDisplaySlot + slot]);
        }
        return res;
    }
    if (val_ptr >= &values[FirstSlot] && val_ptr < &values[int(FirstSlot) + ControlsSlotCount]) {
        const int slot = int(val_ptr - &values[FirstSlot]);
        slot_rows[slot] = row;
        // If the page's "Defaults" button was pressed, it sets values[FirstSlot + slot] = DefaultResetSentinel.
        if (values[FirstSlot + slot] == DefaultResetSentinel) {
            ResetControlsPageToDefaults();
            return 0;
        }
        const int item_idx = std::clamp(controls_scroll + slot, 0, TotalControlsItems - 1);
        const u8 in = u8(a1);
        const bool is_focused = (slot == GetSelectedControlsRow());
        const int k = ItemKeyAction(item_idx);
        if (k >= 0) {
            const int active_k = capturing_action.load(std::memory_order_relaxed);
            if (active_k == k && capturing_slot == slot) {
                StepKeyCapture(row, slot, k);
                return 0;
            }
            if (active_k < 0 && !InCaptureCooldown() && is_focused) {
                bool right = (in != 0) && (HorizontalMenuStep(in) > 0);
                bool del_pressed = false;
                if (SDL_WasInit(SDL_INIT_VIDEO)) {
                    int num_keys = 0;
                    const bool* keys = SDL_GetKeyboardState(&num_keys);
                    if (keys) {
                        if (!right && num_keys > SDL_SCANCODE_RIGHT && keys[SDL_SCANCODE_RIGHT]) {
                            right = true;
                        }
                        if (num_keys > SDL_SCANCODE_DELETE && keys[SDL_SCANCODE_DELETE]) {
                            del_pressed = true;
                        }
                    }
                }
                if (del_pressed) {
                    BbSettings::SetKeyBinding(k, "");
                    BbSettings::Save();
                    RefreshControlsPage();
                    PlayMenuClickSound();
                    capture_cooldown_ms.store(NowSteadyMs() + 300, std::memory_order_relaxed);
                    return 0;
                }
                const bool confirm =
                    !right && (in != 0) && Game<bool (PS4_SYSV_ABI*)(const u8*)>(IsConfirmPressed)(&in);
                if (right || confirm) {
                    PlayMenuClickSound();
                    captured_tokens.clear();
                    captured_wheel.store(0, std::memory_order_relaxed);
                    last_capture_preview.clear();
                    capture_phase = CapturePhase::WaitInitialRelease;
                    last_capture_step_ms.store(NowSteadyMs(), std::memory_order_relaxed);
                    capturing_slot = slot;
                    capturing_alt = right;
                    capturing_action.store(k, std::memory_order_relaxed);
                    ShowCapturePrompt(row, slot, k, {});
                }
            }
            return 0;
        }
        if (item_idx >= ItemMouseSpeed && item_idx <= ItemCameraSens) {
            // Slider row (Mouse Speed, Camera Sensitivity, Controller Vibration)
            SyncSlotSliderWidget(row, slot, is_focused);
            if (is_focused && in != 0 && capturing_action.load(std::memory_order_relaxed) < 0 && !InCaptureCooldown()) {
                const int step = HorizontalMenuStep(in);
                if (step != 0) {
                    StepSlotSetting(row, slot, step < 0 ? -1 : 1, false);
                }
            }
            return 0;
        }
        if (item_idx == ItemCameraFollow) {
            // 3-choice Camera Follow Mode row: supports both native pop-up dropdown (Confirm) and quick Left/Right step.
            SyncSlotSliderWidget(row, slot, is_focused);
            if (row[0x3c0] != 0) {
                const u64 res = row_input_original(row, a1, a2, a3, a4, a5, frame_time);
                if (row[0x3c0] == 0) {
                    ApplyControlsSlotChoiceIndex(row, slot, values[FirstSlot + slot]);
                }
                return res;
            }
            if (is_focused && in != 0 && capturing_action.load(std::memory_order_relaxed) < 0 && !InCaptureCooldown()) {
                const int step = HorizontalMenuStep(in);
                if (step != 0) {
                    StepSlotSetting(row, slot, step, false);
                    return 0;
                }
            }
            return row_input_original(row, a1, a2, a3, a4, a5, frame_time);
        }
        // Camera & Target 2-choice toggle row (ItemCameraX..ItemCameraAdjust): uses <| Normal / Inverted |> ComboBox arrows.
        SyncSlotSliderWidget(row, slot, is_focused);
        if (is_focused && in != 0 && capturing_action.load(std::memory_order_relaxed) < 0 && !InCaptureCooldown()) {
            const int step = HorizontalMenuStep(in);
            const bool confirm =
                (step == 0) && Game<bool (PS4_SYSV_ABI*)(const u8*)>(IsConfirmPressed)(&in);
            if (step != 0 || confirm) {
                StepSlotSetting(row, slot, step != 0 ? step : 1, true);
            }
        }
        return 0;
    }
    if (val_ptr >= &values[FirstEffectsSlot] &&
        val_ptr < &values[int(FirstEffectsSlot) + EffectsSlotCount]) {
        const int slot = int(val_ptr - &values[FirstEffectsSlot]);
        effects_slot_rows[slot] = row;
        if (values[FirstEffectsSlot + slot] == DefaultResetSentinel) {
            ResetEffectsPageToDefaults();
            return 0;
        }
        // Effects 2-choice toggle row: uses <| Off / On |> ComboBox arrows.
        const u8 in = u8(a1);
        const bool is_focused = (slot == GetSelectedEffectsRow());
        SyncEffectsSlotSliderWidget(row, slot, is_focused);
        if (is_focused && in != 0) {
            const int step = HorizontalMenuStep(in);
            const bool confirm =
                (step == 0) && Game<bool (PS4_SYSV_ABI*)(const u8*)>(IsConfirmPressed)(&in);
            if (step != 0 || confirm) {
                StepEffectsSlotSetting(row, slot, step != 0 ? step : 1);
            }
        }
        return 0;
    }
    return row_input_original(row, a1, a2, a3, a4, a5, frame_time);
}

using LookupFn = const char16_t* (PS4_SYSV_ABI*)(void*, u32, u32, u32);
using AddFn = void* (PS4_SYSV_ABI*)(void*, void*, void*, void*);
using BindCommandListFn = void* (PS4_SYSV_ABI*)(void*, void*, void*);
using DeserializeOptionsFn = int (PS4_SYSV_ABI*)(BbMenuEngine::CSOptionsData*, void*);
using CommitPageToOptionsFn = void (PS4_SYSV_ABI*)(BbMenuEngine::CSOptionsData*, const void*);
LookupFn lookup_original = nullptr;
AddFn add_original = nullptr;
BindCommandListFn bind_command_list_original = nullptr;
DeserializeOptionsFn deserialize_options_original = nullptr;
CommitPageToOptionsFn commit_page_to_options_original = nullptr;
std::atomic<const void*> screen_sound_text{nullptr};

int PS4_SYSV_ABI DeserializeOptionsHook(BbMenuEngine::CSOptionsData* self, void* stream) {
    const int res = deserialize_options_original(self, stream);
    ReconcileNativeOptionsOnLoad(self);
    return res;
}

void PS4_SYSV_ABI CommitPageToOptionsHook(BbMenuEngine::CSOptionsData* dst, const void* src) {
    BbMenuEngine::CSOptionsData prev{};
    const bool had_dst = (dst != nullptr);
    if (had_dst) {
        prev = *dst;
    }
    commit_page_to_options_original(dst, src);
    if (had_dst) {
        CaptureNativeOptionsFromGame(*dst, &prev);
    }
}

const char16_t* PS4_SYSV_ABI LookupHook(void* repository, u32 table, u32 category, u32 id) {
    if (OurText(id)) {
        // BB_GAME_MENU_TRACE=1: how often the game asks for our texts (per 2 s).
        static const bool count = [] {
            const char* env = std::getenv("BB_GAME_MENU_TRACE");
            return env && env[0] == '1';
        }();
        if (count) {
            static std::atomic<u64> lookups{0};
            static std::atomic<long long> printed{0};
            ++lookups;
            const long long now = std::chrono::duration_cast<std::chrono::seconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count();
            if (now - printed.load() >= 2) {
                printed = now;
                std::printf("Game menu: %llu lookups of the port's texts in 2 s (last id %u)\n",
                            (unsigned long long)lookups.exchange(0), id - IdBase);
            }
        }
        return TextOf(id);
    }
    if (const int p = building_page.load(); p >= 0 && category == TextName) {
        if (id == pages[p].layout->title) return TextOf(pages[p].name);
        if (pages[p].layout->subtitle && id == pages[p].layout->subtitle) return TextOf(pages[p].help);
    }
    if (category == 0xca && id == 100003 && in_controls_page.load(std::memory_order_relaxed) &&
        active_controls_page != nullptr) {
        const int sel = GetSelectedControlsRow();
        if (sel >= 0 && sel < ControlsSlotCount) {
            const int item_idx = std::clamp(controls_scroll + sel, 0, TotalControlsItems - 1);
            if (ItemKeyAction(item_idx) >= 0) {
                BbPrompts::SetControlsGuideActive(true);
                static const std::u16string edit_en = Utf16(
                    "Edit　<img src='KG_R_L' vspace='-8'>：Alt Edit　<img src='KG_R_U' vspace='-8'>：Clear");
                static const std::u16string edit_ru = Utf16(
                    "Изменить　<img src='KG_R_L' vspace='-8'>：2-я кл.　<img src='KG_R_U' vspace='-8'>：Убрать");
                return RussianTexts() ? edit_ru.c_str() : edit_en.c_str();
            }
        }
    }
    const char16_t* text = lookup_original(repository, table, category, id);
    // BB_MSG_TRACE=1 (diagnostics): the game's text lookups per 2 s by category, with an example
    // id and caller and its text (which menu is on screen, issue #255).
    static const bool msg_trace = [] {
        const char* env = std::getenv("BB_MSG_TRACE");
        return env && env[0] == '1';
    }();
    if (msg_trace) {
        struct Seen {
            u64 count = 0;
            u32 id = 0;
            u64 caller = 0;
            std::string text;
        };
        static std::mutex mutex;
        static std::map<std::pair<u32, u32>, Seen> seen;
        static long long printed = 0;
        std::scoped_lock lock{mutex};
        auto& s = seen[{table, category}];
        if (s.count++ == 0) {
            s.id = id;
            s.caller = reinterpret_cast<u64>(__builtin_return_address(0)) - image_base;
            for (const char16_t* c = text; c && *c && s.text.size() < 40; ++c) {
                s.text += *c < 0x80 ? char(*c) : '?';
            }
        }
        const long long now = std::chrono::duration_cast<std::chrono::seconds>(
                                  std::chrono::steady_clock::now().time_since_epoch())
                                  .count();
        if (now - printed >= 2) {
            printed = now;
            std::string line;
            for (const auto& [key, v] : seen) {
                line += fmt::format(" [{}/{:#x}: {} e.g. {} from {:#x} \"{}\"]", key.first, key.second,
                                    v.count, v.id, v.caller, v.text);
            }
            std::printf("Msg trace (2 s):%s\n", line.c_str());
            seen.clear();
        }
    }
    static const bool trace = [] {
        const char* env = std::getenv("BB_GAME_MENU_TRACE");
        return env && env[0] == '1';
    }();
    if (trace && id >= 110000 && id < 120000) {
        std::printf("Game menu: text %#x/%u%s from %#llx\n", category, id, building_page >= 0 ? " (our page)" : "",
                    (unsigned long long)(reinterpret_cast<u64>(__builtin_return_address(0)) - image_base));
    }
    if (category == TextName && (id == ScreenSoundItem || id == TitleNewGameItem) && text) {
        if (id == ScreenSoundItem) {
            screen_sound_text = text;
        }
        bool cyrillic = false;
        for (const char16_t* c = text; *c; ++c) {
            cyrillic = cyrillic || (*c >= 0x400 && *c < 0x500);
        }
        game_language = cyrillic ? GameLanguage::Russian : GameLanguage::English;
    }
    return text;
}

void AddItem(void* menu, u32 name, u32 help, void* factory) {
    using ToFunction = void* (PS4_SYSV_ABI*)(void*, void*);
    alignas(16) u8 pair[2 * TextSize] = {};
    MakeTexts(pair, name, help);
    alignas(16) u8 source[FunctionSize] = {};
    *reinterpret_cast<u64*>(source) = image_base + FactoryVtable;
    *reinterpret_cast<void**>(source + 8) = factory;
    *reinterpret_cast<void**>(source + 0x20) = source;
    alignas(16) u8 function[FunctionSize] = {};
    *reinterpret_cast<void**>(function + 0x20) = Game<ToFunction>(FactoryToFunction)(source, function);
    alignas(16) u8 out[16] = {};
    add_original(menu, pair, function, out);
    DestroyFunction(function);
    DestroyTexts(pair);
}

void* PS4_SYSV_ABI AddHook(void* menu, void* pair, void* function, void* out) {
    void* result = add_original(menu, pair, function, out);
    const void* item_text = *reinterpret_cast<const void* const*>(pair);
    const void* screen_sound = screen_sound_text.load();
    // Place Display and Effects right under Environment / Screen & Sound (110001).
    // Controls (110000) itself now houses all Keybinds + Mouse + Camera settings in one window.
    if (screen_sound && item_text == screen_sound) {
        for (std::size_t i = 0; i < pages.size() && i < MaxPages; ++i) {
            AddItem(menu, pages[i].name, pages[i].help, factories[i]);
        }
    }
    return result;
}

void* PS4_SYSV_ABI ExitGameFactory(void* out, void* /*arg*/) {
    BbGameMenu::FlushPendingSaves();
    std::printf("Game menu: Exit Game selected by user\n");
    std::fflush(stdout);
    std::fflush(stderr);
    *static_cast<u64*>(out) = 0;
    SDL_Event quit{};
    quit.type = SDL_EVENT_QUIT;
    SDL_PushEvent(&quit);
    std::_Exit(0);
    return out;
}

void* PS4_SYSV_ABI BindCommandListHook(void* out, void* menu, void* cmd_list_mc) {
    const u64 ret_va = reinterpret_cast<u64>(__builtin_return_address(0)) - image_base;
    if (ret_va == TitleMenuBindRet && exit_game_name != 0) {
        AddItem(menu, exit_game_name, exit_game_help, reinterpret_cast<void*>(&ExitGameFactory));
    }
    return bind_command_list_original(out, menu, cmd_list_mc);
}

/// `jmp hook` at the function (via a near thunk); its moved prologue + `jmp` back is the original.
bool Detour(unsigned char* image, u64 va, const u8* prologue, std::size_t length, void* hook,
            void** original, u8*& cursor) {
    u8* site = image + va;
    if (std::memcmp(site, prologue, length) != 0) {
        std::printf("Game menu: unexpected code at %#llx, not hooked\n", (unsigned long long)va);
        return false;
    }
    u8* trampoline = cursor;
    std::memcpy(trampoline, prologue, length);
    trampoline[length] = 0xe9;
    const std::int64_t back = std::int64_t(site + length) - std::int64_t(trampoline + length + 5);
    const std::int32_t back32 = std::int32_t(back);
    std::memcpy(trampoline + length + 1, &back32, 4);
    u8* thunk = trampoline + 32;
    thunk[0] = 0x48;
    thunk[1] = 0xb8; // movabs rax, hook
    std::memcpy(thunk + 2, &hook, 8);
    thunk[10] = 0xff;
    thunk[11] = 0xe0; // jmp rax
    const std::int64_t to = std::int64_t(thunk) - std::int64_t(site + 5);
    if (back != back32 || to != std::int32_t(to)) {
        std::printf("Game menu: stubs out of jump range, not hooked\n");
        return false;
    }
    const std::int32_t to32 = std::int32_t(to);
    const u64 page = u64(sysconf(_SC_PAGESIZE));
    const u64 patch_start = reinterpret_cast<u64>(site) & ~(page - 1);
    mprotect(reinterpret_cast<void*>(patch_start),
             reinterpret_cast<u64>(site) + length - patch_start,
             PROT_READ | PROT_WRITE | PROT_EXEC);
    site[0] = 0xe9;
    std::memcpy(site + 1, &to32, 4);
    for (std::size_t i = 5; i < length; ++i) {
        site[i] = 0xcc;
    }
    *original = trampoline;
    cursor += 64;
    return true;
}

/// A mapping within +-2 GiB of the image, for rel32 jumps both ways.
u8* MapNear(unsigned char* image, u64 image_size, std::size_t size) {
    const u64 base = reinterpret_cast<u64>(image);
    const u64 above_base = (base + image_size + 0xFFFFull) & ~0xFFFFull;
    for (u64 k = 1; k <= 64; ++k) {
        for (const u64 hint : {base - k * (24ull << 20), above_base + k * (24ull << 20)}) {
            void* p = mmap(reinterpret_cast<void*>(hint & ~0xFFFFull), size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            if (p != MAP_FAILED) {
                return static_cast<u8*>(p);
            }
        }
    }
    return nullptr;
}

} // namespace

void PatchImage(unsigned char* image, std::uint64_t size) {
    if (const char* env = std::getenv("BB_GAME_MENU"); env && env[0] == '0') {
        return;
    }
    image_base = reinterpret_cast<u64>(image);
    const std::size_t page = std::size_t(sysconf(_SC_PAGESIZE));
    u8* stubs = MapNear(image, size, page);
    if (!stubs) {
        std::printf("Game menu: no memory near the image, the port's pages are off\n");
        return;
    }
    DefineTexts();
    for (const Page& p : pages) {
        for (const Row& r : p.rows) {
            byte_field[r.field] = r.kind == Kind::Toggle || r.kind == Kind::Slider;
        }
    }
    static const BbSettings::Values defaults;
    Fill(defaults, default_values);
    for (int e = 0; e < BbSettings::EffectCount; ++e) {
        default_values[FirstEffect + e] = BbSettings::Effects[e].default_on;
    }
    for (int slot = 0; slot < ControlsSlotCount; ++slot) {
        default_values[FirstSlot + slot] = DefaultResetSentinel; // triggers default reset when "Defaults" is clicked
    }
    for (int slot = 0; slot < DisplaySlotCount; ++slot) {
        default_values[FirstDisplaySlot + slot] = DefaultResetSentinel; // triggers default reset on Display page
    }
    for (int slot = 0; slot < EffectsSlotCount; ++slot) {
        default_values[FirstEffectsSlot + slot] = DefaultResetSentinel; // triggers default reset on Effects page
    }
    for (int f = 0; f < FieldCount; ++f) {
        default_bytes[f] = u8(default_values[f]);
    }
    u8* cursor = stubs;
    // The lookup first: the menu hook only acts once it has seen the Screen/Sound text.
    const bool ok =
        Detour(image, MsgLookup, MsgLookupPrologue, sizeof(MsgLookupPrologue),
               reinterpret_cast<void*>(&LookupHook), reinterpret_cast<void**>(&lookup_original), cursor) &&
        Detour(image, AddPageItem, AddPageItemPrologue, sizeof(AddPageItemPrologue),
               reinterpret_cast<void*>(&AddHook), reinterpret_cast<void**>(&add_original), cursor) &&
        Detour(image, ControlsContent, AddPageItemPrologue, sizeof(AddPageItemPrologue),
               reinterpret_cast<void*>(&ControlsContentHook),
               reinterpret_cast<void**>(&controls_content_original), cursor) &&
        Detour(image, OptionsPageNavigate, AddPageItemPrologue, sizeof(AddPageItemPrologue),
               reinterpret_cast<void*>(&OptionsPageNavigateHook),
               reinterpret_cast<void**>(&options_page_navigate_original), cursor) &&
        Detour(image, OptionsPageResetDefaults, AddPageItemPrologue, sizeof(AddPageItemPrologue),
               reinterpret_cast<void*>(&OptionsPageResetDefaultsHook),
               reinterpret_cast<void**>(&options_page_reset_defaults_original), cursor) &&
        Detour(image, RowInput, RowInputPrologue, sizeof(RowInputPrologue),
               reinterpret_cast<void*>(&RowInputHook), reinterpret_cast<void**>(&row_input_original), cursor) &&
        Detour(image, BindCommandList, AddPageItemPrologue, sizeof(AddPageItemPrologue),
               reinterpret_cast<void*>(&BindCommandListHook),
               reinterpret_cast<void**>(&bind_command_list_original), cursor) &&
        Detour(image, BbMenuEngine::Vma::DeserializeOptionsData, AddPageItemPrologue,
               sizeof(AddPageItemPrologue), reinterpret_cast<void*>(&DeserializeOptionsHook),
               reinterpret_cast<void**>(&deserialize_options_original), cursor) &&
        Detour(image, BbMenuEngine::Vma::CommitPageToOptions, AddPageItemPrologue,
               sizeof(AddPageItemPrologue), reinterpret_cast<void*>(&CommitPageToOptionsHook),
               reinterpret_cast<void**>(&commit_page_to_options_original), cursor);
    mprotect(stubs, page, PROT_READ | PROT_EXEC);
    std::printf("Game menu: %s (Title Exit Game + unified Controls + %zu pages in System)\n",
                ok ? "the port's pages are in" : "hooks failed, off", pages.size());
}

bool CapturesInput() {
    const long long now = NowSteadyMs();
    if (capturing_action.load(std::memory_order_relaxed) >= 0) {
        if (now - last_capture_step_ms.load(std::memory_order_relaxed) > 400) {
            capturing_action.store(-1, std::memory_order_relaxed);
            return false;
        }
        return true;
    }
    return false;
}

bool InCaptureCooldown() {
    return NowSteadyMs() < capture_cooldown_ms.load(std::memory_order_relaxed);
}

void NotifyWheelCapture(int dir) {
    if (capturing_action.load(std::memory_order_relaxed) >= 0 && dir != 0) {
        captured_wheel.store(dir > 0 ? 1 : -1, std::memory_order_relaxed);
    }
}

bool DirectWheelScrollActive() {
    return NowSteadyMs() - last_direct_wheel_ms.load(std::memory_order_relaxed) < 150;
}

bool CommitDebouncedDisplaySettings(BbSettings::Values& s, bool force) {
    if (!display_heavy_dirty.load(std::memory_order_acquire)) {
        return false;
    }
    if (!force && NowSteadyMs() - display_heavy_dirty_ms.load(std::memory_order_relaxed) < 100) {
        return false;
    }
    display_heavy_dirty.store(false, std::memory_order_release);
    const int aspect = std::clamp(disp_aspect, 0, BbSettings::AspectCount - 1);
    const auto chosen_out = BbSettings::PackSize(MenuOutput(aspect, disp_output_idx));
    if (chosen_out != s.output_res.load()) {
        s.output_res_auto = false;
        s.output_res = chosen_out;
    }
    s.aspect = aspect;

    int target_upscaler = BbSettings::UpscalerOff;
    int target_dlss_preset = s.dlss_preset.load();
    const int target_preset = std::clamp(disp_upscale_preset, int(BbSettings::NativeAA), int(BbSettings::UltraPerformance));
    const bool target_fxaa = (disp_aa_mode == AaFxaa);

    if (disp_upscaler_choice == DispUpOff) {
        target_upscaler = BbSettings::UpscalerOff;
        if (disp_aa_mode == AaDlaa) {
            target_dlss_preset = 11;
        } else if (disp_aa_mode == AaDlaaCnn) {
            target_dlss_preset = 6;
        }
    } else {
        switch (disp_upscaler_choice) {
        case DispUpDlss4:
            target_upscaler = BbSettings::UpscalerDlss;
            target_dlss_preset = 11;
            break;
        case DispUpDlss45M:
            target_upscaler = BbSettings::UpscalerDlss;
            target_dlss_preset = 13;
            break;
        case DispUpDlss45L:
            target_upscaler = BbSettings::UpscalerDlss;
            target_dlss_preset = 12;
            break;
        case DispUpDlss3:
            target_upscaler = BbSettings::UpscalerDlss;
            target_dlss_preset = 6;
            break;
        case DispUpFsr4:
            target_upscaler = BbSettings::UpscalerFsr4;
            break;
        case DispUpFsr411:
            target_upscaler = BbSettings::UpscalerFsr411;
            break;
        case DispUpFsr31:
        default:
            target_upscaler = BbSettings::UpscalerFsr3;
            break;
        }
    }

    s.aa_mode = disp_aa_mode;
    s.aa_mode_chosen = true;
    s.upscaler = UpscalerAvailable(target_upscaler) ? target_upscaler : int(BbSettings::UpscalerFsr3);
    s.dlss_preset = target_dlss_preset;
    s.preset = target_preset;
    s.effects[GameAaEffectIdx] = target_fxaa;
    values[FirstEffect + GameAaEffectIdx] = target_fxaa ? 1 : 0;
    applied[FirstEffect + GameAaEffectIdx] = target_fxaa ? 1 : 0;
    bytes[FirstEffect + GameAaEffectIdx] = u8(target_fxaa ? 1 : 0);
    BbGuestHooks::ApplyGameEffects();
    return true;
}

void FlushPendingSaves() {
    auto& s = BbSettings::Get();
    if (values_valid && CommitDebouncedDisplaySettings(s, true)) {
        BbSettings::Save();
    } else {
        BbSettings::Save();
    }
    if (auto* opt = ActiveSystemOptions()) {
        QueueNativeOptionsDiskSync(*opt);
    }
    FlushNativeOptionsDiskSync(true);
    runtime_file_drain_sync();
}

void Poll() {
    const long long now_ms = NowSteadyMs();
    if (!native_options_bootstrapped.load(std::memory_order_acquire) && image_base != 0) {
        u8* const slm = *reinterpret_cast<u8* const*>(image_base + BbMenuEngine::Vma::FrpgSaveLoadManPtr);
        if (slm && slm[BbMenuEngine::FrpgSaveLoadManLayout::kSaveEnabled] != 0) {
            if (auto* opt = ActiveSystemOptions()) {
                ReconcileNativeOptionsOnLoad(opt);
            }
        }
    }

    if (now_ms - last_external_ini_poll_ms.load(std::memory_order_relaxed) >= 400) {
        last_external_ini_poll_ms.store(now_ms, std::memory_order_relaxed);
        if (BbSettings::ReloadExternalIfModified()) {
            BbGuestHooks::ApplyGameEffects();
            if (auto* opt = ActiveSystemOptions()) {
                ApplySettingsToLiveNativeOptions(opt);
            }
        }
    }

    if (native_options_dirty.load(std::memory_order_acquire) &&
        now_ms - native_options_dirty_ms.load(std::memory_order_relaxed) >= 150) {
        FlushNativeOptionsDiskSync(false);
    }

    if (!values_valid) {
        return;
    }
    auto& s = BbSettings::Get();
    bool saved = CommitDebouncedDisplaySettings(s, false);

    // Apply Effects page toggles immediately.
    int now_values[FieldCount];
    std::memcpy(now_values, values, sizeof(now_values));
    for (int f = 0; f < FieldCount; ++f) {
        if (byte_field[f]) {
            now_values[f] = bytes[f];
        }
    }
    bool effects_changed = false;
    for (int e = 0; e < BbSettings::EffectCount; ++e) {
        if (now_values[FirstEffect + e] != applied[FirstEffect + e]) {
            s.effects[e] = now_values[FirstEffect + e] != 0;
            applied[FirstEffect + e] = now_values[FirstEffect + e];
            if (e == GameAaEffectIdx && disp_upscaler_choice == DispUpOff &&
                (disp_aa_mode == AaOff || disp_aa_mode == AaFxaa)) {
                disp_aa_mode = s.effects[e].load() ? AaFxaa : AaOff;
            }
            effects_changed = true;
        }
    }
    if (effects_changed) {
        BbGuestHooks::ApplyGameEffects();
        saved = true;
    }
    if (saved) {
        BbSettings::Save();
    }
}

} // namespace BbGameMenu
