// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.features.settings.model

enum class IntSetting(
    override val key: String,
    override val section: String,
    override val defaultValue: Int
) : AbstractIntSetting {
    FRAME_LIMIT("frame_limit", Settings.SECTION_RENDERER, 100),
    EMULATED_REGION("region_value", Settings.SECTION_SYSTEM, -1),
    INIT_CLOCK("init_clock", Settings.SECTION_SYSTEM, 0),
    INIT_TICKS_TYPE("init_ticks_type", Settings.SECTION_SYSTEM, 0),
    INIT_TICKS_OVERRIDE("init_ticks_override", Settings.SECTION_SYSTEM, 0),
    CAMERA_INNER_FLIP("camera_inner_flip", Settings.SECTION_CAMERA, 0),
    CAMERA_OUTER_LEFT_FLIP("camera_outer_left_flip", Settings.SECTION_CAMERA, 0),
    CAMERA_OUTER_RIGHT_FLIP("camera_outer_right_flip", Settings.SECTION_CAMERA, 0),
    GRAPHICS_API("graphics_api", Settings.SECTION_RENDERER, 1),
    RESOLUTION_FACTOR("resolution_factor", Settings.SECTION_RENDERER, 1),
    STEREOSCOPIC_3D_MODE("render_3d", Settings.SECTION_RENDERER, 0),
    STEREOSCOPIC_3D_DEPTH("factor_3d", Settings.SECTION_RENDERER, 0),
    MONO_RENDER_OPTION("mono_render_option", Settings.SECTION_RENDERER, 0),
    STEPS_PER_HOUR("steps_per_hour", Settings.SECTION_SYSTEM, 0),
    CARDBOARD_SCREEN_SIZE("cardboard_screen_size", Settings.SECTION_LAYOUT, 85),
    CARDBOARD_X_SHIFT("cardboard_x_shift", Settings.SECTION_LAYOUT, 0),
    CARDBOARD_Y_SHIFT("cardboard_y_shift", Settings.SECTION_LAYOUT, 0),
    SCREEN_LAYOUT("layout_option", Settings.SECTION_LAYOUT, 0),
    SMALL_SCREEN_POSITION("small_screen_position", Settings.SECTION_LAYOUT, 2),
    LANDSCAPE_TOP_X("custom_top_x", Settings.SECTION_LAYOUT, 0),
    LANDSCAPE_TOP_Y("custom_top_y", Settings.SECTION_LAYOUT, 0),
    LANDSCAPE_TOP_WIDTH("custom_top_width", Settings.SECTION_LAYOUT, 800),
    LANDSCAPE_TOP_HEIGHT("custom_top_height", Settings.SECTION_LAYOUT, 480),
    LANDSCAPE_BOTTOM_X("custom_bottom_x", Settings.SECTION_LAYOUT, 80),
    LANDSCAPE_BOTTOM_Y("custom_bottom_y", Settings.SECTION_LAYOUT, 480),
    LANDSCAPE_BOTTOM_WIDTH("custom_bottom_width", Settings.SECTION_LAYOUT, 640),
    LANDSCAPE_BOTTOM_HEIGHT("custom_bottom_height", Settings.SECTION_LAYOUT, 480),
    PORTRAIT_SCREEN_LAYOUT("portrait_layout_option", Settings.SECTION_LAYOUT, 0),
    PORTRAIT_TOP_X("custom_portrait_top_x", Settings.SECTION_LAYOUT, 0),
    PORTRAIT_TOP_Y("custom_portrait_top_y", Settings.SECTION_LAYOUT, 0),
    PORTRAIT_TOP_WIDTH("custom_portrait_top_width", Settings.SECTION_LAYOUT, 800),
    PORTRAIT_TOP_HEIGHT("custom_portrait_top_height", Settings.SECTION_LAYOUT, 480),
    PORTRAIT_BOTTOM_X("custom_portrait_bottom_x", Settings.SECTION_LAYOUT, 80),
    PORTRAIT_BOTTOM_Y("custom_portrait_bottom_y", Settings.SECTION_LAYOUT, 480),
    PORTRAIT_BOTTOM_WIDTH("custom_portrait_bottom_width", Settings.SECTION_LAYOUT, 640),
    PORTRAIT_BOTTOM_HEIGHT("custom_portrait_bottom_height", Settings.SECTION_LAYOUT, 480),
    AUDIO_EMULATION("audio_emulation", Settings.SECTION_AUDIO, 0),
    AUDIO_INPUT_TYPE("input_type", Settings.SECTION_AUDIO, 0),
    AUDIO_OUTPUT_TYPE("output_type", Settings.SECTION_AUDIO, 0),
    CPU_CLOCK_SPEED("cpu_clock_percentage", Settings.SECTION_CORE, 100),
    GDB_PORT("gdbstub_port", Settings.SECTION_DEBUG, 24689),
    TEXTURE_FILTER("texture_filter", Settings.SECTION_RENDERER, 0),
    TEXTURE_SAMPLING("texture_sampling", Settings.SECTION_RENDERER, 0),
    CUSTOM_CPU_TICKS("custom_cpu_ticks", Settings.SECTION_CORE, 16000),
    OPTIMIZE_SPIRV("optimize_spirv_output", Settings.SECTION_RENDERER, 0),
    FRAME_SKIP("frame_skip", Settings.SECTION_RENDERER, 0),
    DELAY_RENDER_THREAD_US("delay_game_render_thread_us", Settings.SECTION_RENDERER, 0),
    ORIENTATION_OPTION("screen_orientation", Settings.SECTION_LAYOUT, 2),
    TURBO_SPEED("turbo_speed", Settings.SECTION_CORE, 200),
    PERF_OVERLAY_POSITION("perf_overlay_position", Settings.SECTION_LAYOUT, 0);

    override var int: Int = defaultValue

    override val valueAsString: String
        get() = int.toString()

    override val isRuntimeEditable: Boolean
        get() {
            for (setting in NOT_RUNTIME_EDITABLE) {
                if (setting == this) {
                    return false
                }
            }
            return true
        }

    companion object {
        private val NOT_RUNTIME_EDITABLE = listOf(
            EMULATED_REGION,
            INIT_CLOCK,
            GRAPHICS_API,
            AUDIO_INPUT_TYPE,
            FRAME_LIMIT
        )

        fun from(key: String): IntSetting? = IntSetting.entries.firstOrNull { it.key == key }

        fun clear() = IntSetting.entries.forEach { it.int = it.defaultValue }
    }
}
