// Copyright 2016 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <QColorDialog>
#include <QStandardItemModel>
#include "borked3ds_qt/configuration/configuration_shared.h"
#include "borked3ds_qt/configuration/configure_graphics.h"
#include "common/settings.h"
#include "ui_configure_graphics.h"
#ifdef ENABLE_VULKAN
#include "video_core/renderer_vulkan/vk_instance.h"
#endif

ConfigureGraphics::ConfigureGraphics(QString gl_renderer, std::span<const QString> physical_devices,
                                     bool is_powered_on, QWidget* parent)
    : QWidget(parent), ui(std::make_unique<Ui::ConfigureGraphics>()) {
    ui->setupUi(this);

    SetupPerGameUI();

    ui->graphics_api_combo->setEnabled(!is_powered_on);
    ui->physical_device_combo->setEnabled(!is_powered_on);
    ui->toggle_async_shaders->setEnabled(!is_powered_on);
    ui->toggle_async_present->setEnabled(!is_powered_on);
    ui->toggle_accurate_mul->setEnabled(!is_powered_on);
    // Set the index to -1 to ensure the below lambda is called with setCurrentIndex
    ui->graphics_api_combo->setCurrentIndex(-1);

    const auto width = static_cast<int>(QString::fromStdString("000000000").size() * 6);
    ui->delay_render_display_label->setMinimumWidth(width);
    ui->delay_render_combo->setVisible(!Settings::IsConfiguringGlobal());

    auto graphics_api_combo_model =
        qobject_cast<QStandardItemModel*>(ui->graphics_api_combo->model());
#ifndef ENABLE_SOFTWARE_RENDERER
    const auto software_item =
        graphics_api_combo_model->item(static_cast<u32>(Settings::GraphicsAPI::Software));
    software_item->setFlags(software_item->flags() & ~Qt::ItemIsEnabled);
#endif

#ifndef ENABLE_OPENGL
    const auto opengl_item =
        graphics_api_combo_model->item(static_cast<u32>(Settings::GraphicsAPI::OpenGL));
    opengl_item->setFlags(opengl_item->flags() & ~Qt::ItemIsEnabled);
#else
    ui->opengl_renderer_name_label->setText(gl_renderer);
#endif

#ifndef ENABLE_VULKAN
    const auto vulkan_item =
        graphics_api_combo_model->item(static_cast<u32>(Settings::GraphicsAPI::Vulkan));
    vulkan_item->setFlags(vulkan_item->flags() & ~Qt::ItemIsEnabled);
#else
    if (physical_devices.empty()) {
        const auto vulkan_item =
            graphics_api_combo_model->item(static_cast<u32>(Settings::GraphicsAPI::Vulkan));
        vulkan_item->setFlags(vulkan_item->flags() & ~Qt::ItemIsEnabled);

        ui->physical_device_combo->setVisible(false);
        ui->spirv_shader_gen->setVisible(false);
        ui->toggle_geometry_shader->setVisible(false);
        ui->sample_shading->setVisible(false);
        ui->toggle_relaxed_precision->setVisible(false);
        ui->widget_optimize_spirv->setVisible(false);
    } else {
        for (const QString& name : physical_devices) {
            ui->physical_device_combo->addItem(name);
        }
    }
#endif

    connect(ui->graphics_api_combo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                const auto graphics_api =
                    ConfigurationShared::GetComboboxSetting(index, &Settings::values.graphics_api);
                const bool is_software = graphics_api == Settings::GraphicsAPI::Software;

                ui->hw_renderer_group->setEnabled(!is_software);
                ui->toggle_disk_shader_cache->setEnabled(!is_software &&
                                                         ui->toggle_hw_shader->isChecked());
            });

    connect(ui->toggle_hw_shader, &QCheckBox::toggled, this, [this] {
        const bool enabled = ui->toggle_hw_shader->isEnabled();
        const bool checked = ui->toggle_hw_shader->isChecked();
        ui->hw_shader_group->setEnabled(checked && enabled);
        ui->toggle_disk_shader_cache->setEnabled(checked && enabled);
    });

    connect(ui->spirv_shader_gen, &QCheckBox::toggled, this, [this] {
        const bool enabled = ui->spirv_shader_gen->isEnabled();
        const bool checked = ui->spirv_shader_gen->isChecked();
        ui->toggle_relaxed_precision->setEnabled(checked && enabled);
        ui->widget_optimize_spirv->setEnabled(checked && enabled);
    });

    connect(ui->optimize_spirv_combobox, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int index) {
                const auto spirv_opt = ConfigurationShared::GetComboboxSetting(
                    index, &Settings::values.optimize_spirv_output);
                const bool is_active = spirv_opt != Settings::OptimizeSpirv::Disabled;
                const bool enabled = ui->spirv_shader_gen->isEnabled();
                const bool checked = ui->spirv_shader_gen->isChecked();

                ui->toggle_spirv_validation->setEnabled(enabled && checked && is_active);
                ui->toggle_spirv_legalization->setEnabled(enabled && checked && is_active);
            });

    connect(ui->graphics_api_combo, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &ConfigureGraphics::SetPhysicalDeviceComboVisibility);

    ui->toggle_spirv_validation->setEnabled(Settings::values.optimize_spirv_output.GetValue() !=
                                            Settings::OptimizeSpirv::Disabled);
    ui->toggle_spirv_legalization->setEnabled(Settings::values.optimize_spirv_output.GetValue() !=
                                              Settings::OptimizeSpirv::Disabled);

    connect(ui->delay_render_slider, &QSlider::valueChanged, this, [&](int value) {
        ui->delay_render_display_label->setText(
            QStringLiteral("%1 ms")
                .arg(((double)value) / 1000.f, 0, 'f', 3)
                .rightJustified(QString::fromStdString("000000000").size()));
    });

    SetConfiguration();
}

ConfigureGraphics::~ConfigureGraphics() = default;

void ConfigureGraphics::SetConfiguration() {
    ui->delay_render_slider->setValue(Settings::values.delay_game_render_thread_us.GetValue());
    ui->delay_render_display_label->setText(
        QStringLiteral("%1 ms")
            .arg(((double)ui->delay_render_slider->value()) / 1000, 0, 'f', 3)
            .rightJustified(QString::fromStdString("000000000").size()));

    if (!Settings::IsConfiguringGlobal()) {
        ConfigurationShared::SetHighlight(ui->graphics_api_group,
                                          !Settings::values.graphics_api.UsingGlobal());
        ConfigurationShared::SetPerGameSetting(ui->graphics_api_combo,
                                               &Settings::values.graphics_api);
        ConfigurationShared::SetHighlight(ui->physical_device_group,
                                          !Settings::values.physical_device.UsingGlobal());
        ConfigurationShared::SetPerGameSetting(ui->physical_device_combo,
                                               &Settings::values.physical_device);
        ConfigurationShared::SetHighlight(ui->widget_optimize_spirv,
                                          !Settings::values.optimize_spirv_output.UsingGlobal());
        ConfigurationShared::SetPerGameSetting(ui->optimize_spirv_combobox,
                                               &Settings::values.optimize_spirv_output);
        ConfigurationShared::SetHighlight(ui->widget_texture_sampling,
                                          !Settings::values.texture_sampling.UsingGlobal());
        ConfigurationShared::SetPerGameSetting(ui->texture_sampling_combobox,
                                               &Settings::values.texture_sampling);
        ConfigurationShared::SetHighlight(
            ui->delay_render_layout, !Settings::values.delay_game_render_thread_us.UsingGlobal());

        if (Settings::values.delay_game_render_thread_us.UsingGlobal()) {
            ui->delay_render_combo->setCurrentIndex(0);
            ui->delay_render_slider->setEnabled(false);
        } else {
            ui->delay_render_combo->setCurrentIndex(1);
            ui->delay_render_slider->setEnabled(true);
        }
    } else {
        ui->graphics_api_combo->setCurrentIndex(
            static_cast<int>(Settings::values.graphics_api.GetValue()));
        ui->physical_device_combo->setCurrentIndex(
            static_cast<int>(Settings::values.physical_device.GetValue()));
        ui->optimize_spirv_combobox->setCurrentIndex(
            static_cast<int>(Settings::values.optimize_spirv_output.GetValue()));
        ui->texture_sampling_combobox->setCurrentIndex(
            static_cast<int>(Settings::values.texture_sampling.GetValue()));
    }

    ui->toggle_use_gles->setChecked(Settings::values.use_gles.GetValue());
    ui->toggle_hw_shader->setChecked(Settings::values.use_hw_shader.GetValue());
    ui->toggle_accurate_mul->setChecked(Settings::values.shaders_accurate_mul.GetValue());
    ui->toggle_disk_shader_cache->setChecked(Settings::values.use_disk_shader_cache.GetValue());
    ui->toggle_vsync_new->setChecked(Settings::values.use_vsync_new.GetValue());
    ui->spirv_shader_gen->setChecked(Settings::values.spirv_shader_gen.GetValue());
    ui->toggle_geometry_shader->setChecked(Settings::values.geometry_shader.GetValue());
    ui->sample_shading->setChecked(Settings::values.use_sample_shading.GetValue());
    ui->toggle_relaxed_precision->setChecked(
        Settings::values.relaxed_precision_decorators.GetValue());
    ui->toggle_spirv_validation->setChecked(Settings::values.spirv_output_validation.GetValue());
    ui->toggle_spirv_legalization->setChecked(
        Settings::values.spirv_output_legalization.GetValue());
    ui->toggle_async_shaders->setChecked(Settings::values.async_shader_compilation.GetValue());
    ui->toggle_async_present->setChecked(Settings::values.async_presentation.GetValue());
    ui->toggle_skip_slow_draw->setChecked(Settings::values.skip_slow_draw.GetValue());
    ui->toggle_skip_texture_copy->setChecked(Settings::values.skip_texture_copy.GetValue());
    ui->toggle_skip_cpu_write->setChecked(Settings::values.skip_cpu_write.GetValue());
    ui->toggle_upscaling_hack->setChecked(Settings::values.upscaling_hack.GetValue());

    ui->toggle_spirv_validation->setEnabled(Settings::values.optimize_spirv_output.GetValue() !=
                                            Settings::OptimizeSpirv::Disabled);
    ui->toggle_spirv_legalization->setEnabled(Settings::values.optimize_spirv_output.GetValue() !=
                                              Settings::OptimizeSpirv::Disabled);

    if (Settings::IsConfiguringGlobal()) {
        ui->toggle_shader_jit->setChecked(Settings::values.use_shader_jit.GetValue());
    }
}

void ConfigureGraphics::ApplyConfiguration() {
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.graphics_api,
                                             ui->graphics_api_combo);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.physical_device,
                                             ui->physical_device_combo);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.async_shader_compilation,
                                             ui->toggle_async_shaders, async_shader_compilation);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.use_gles, ui->toggle_use_gles,
                                             use_gles);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.async_presentation,
                                             ui->toggle_async_present, async_presentation);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.skip_slow_draw,
                                             ui->toggle_skip_slow_draw, skip_slow_draw);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.skip_texture_copy,
                                             ui->toggle_skip_texture_copy, skip_texture_copy);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.skip_cpu_write,
                                             ui->toggle_skip_cpu_write, skip_cpu_write);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.upscaling_hack,
                                             ui->toggle_upscaling_hack, upscaling_hack);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.spirv_shader_gen,
                                             ui->spirv_shader_gen, spirv_shader_gen);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.geometry_shader,
                                             ui->toggle_geometry_shader, geometry_shader);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.use_sample_shading,
                                             ui->sample_shading, sample_shading);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.relaxed_precision_decorators,
                                             ui->toggle_relaxed_precision,
                                             relaxed_precision_decorators);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.optimize_spirv_output,
                                             ui->optimize_spirv_combobox);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.spirv_output_validation,
                                             ui->toggle_spirv_validation, spirv_output_validation);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.spirv_output_legalization,
                                             ui->toggle_spirv_legalization,
                                             spirv_output_legalization);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.use_hw_shader, ui->toggle_hw_shader,
                                             use_hw_shader);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.shaders_accurate_mul,
                                             ui->toggle_accurate_mul, shaders_accurate_mul);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.texture_sampling,
                                             ui->texture_sampling_combobox);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.use_disk_shader_cache,
                                             ui->toggle_disk_shader_cache, use_disk_shader_cache);
    ConfigurationShared::ApplyPerGameSetting(&Settings::values.use_vsync_new, ui->toggle_vsync_new,
                                             use_vsync_new);
    ConfigurationShared::ApplyPerGameSetting(
        &Settings::values.delay_game_render_thread_us, ui->delay_render_combo,
        [this](s32) { return ui->delay_render_slider->value(); });

    if (Settings::IsConfiguringGlobal()) {
        Settings::values.use_shader_jit = ui->toggle_shader_jit->isChecked();
    }
}

void ConfigureGraphics::RetranslateUI() {
    ui->retranslateUi(this);
}

void ConfigureGraphics::SetupPerGameUI() {
    // Block the global settings if a game is currently running that overrides them
    if (Settings::IsConfiguringGlobal()) {
        ui->graphics_api_group->setEnabled(Settings::values.graphics_api.UsingGlobal());
        ui->toggle_hw_shader->setEnabled(Settings::values.use_hw_shader.UsingGlobal());
        ui->toggle_accurate_mul->setEnabled(Settings::values.shaders_accurate_mul.UsingGlobal());
        ui->toggle_disk_shader_cache->setEnabled(
            Settings::values.use_disk_shader_cache.UsingGlobal());
        ui->toggle_vsync_new->setEnabled(ui->toggle_vsync_new->isEnabled() &&
                                         Settings::values.use_vsync_new.UsingGlobal());
        ui->toggle_async_shaders->setEnabled(
            Settings::values.async_shader_compilation.UsingGlobal());
        ui->widget_optimize_spirv->setEnabled(Settings::values.optimize_spirv_output.UsingGlobal());
        ui->widget_texture_sampling->setEnabled(Settings::values.texture_sampling.UsingGlobal());
        ui->toggle_async_present->setEnabled(Settings::values.async_presentation.UsingGlobal());
        ui->graphics_api_combo->setEnabled(Settings::values.graphics_api.UsingGlobal());
        ui->physical_device_combo->setEnabled(Settings::values.physical_device.UsingGlobal());
        ui->delay_render_combo->setEnabled(
            Settings::values.delay_game_render_thread_us.UsingGlobal());
        return;
    }

    connect(ui->delay_render_combo, qOverload<int>(&QComboBox::activated), this, [this](int index) {
        ui->delay_render_slider->setEnabled(index == 1);
        ConfigurationShared::SetHighlight(ui->delay_render_layout, index == 1);
    });

    ui->toggle_shader_jit->setVisible(false);

    ConfigurationShared::SetColoredComboBox(
        ui->graphics_api_combo, ui->graphics_api_group,
        static_cast<u32>(Settings::values.graphics_api.GetValue(true)));

    ConfigurationShared::SetColoredComboBox(
        ui->physical_device_combo, ui->physical_device_group,
        static_cast<u32>(Settings::values.physical_device.GetValue(true)));

    ConfigurationShared::SetColoredComboBox(
        ui->optimize_spirv_combobox, ui->widget_optimize_spirv,
        static_cast<int>(Settings::values.optimize_spirv_output.GetValue(true)));

    ConfigurationShared::SetColoredComboBox(
        ui->texture_sampling_combobox, ui->widget_texture_sampling,
        static_cast<int>(Settings::values.texture_sampling.GetValue(true)));

    ConfigurationShared::SetColoredTristate(ui->toggle_hw_shader, Settings::values.use_hw_shader,
                                            use_hw_shader);
    ConfigurationShared::SetColoredTristate(
        ui->toggle_accurate_mul, Settings::values.shaders_accurate_mul, shaders_accurate_mul);
    ConfigurationShared::SetColoredTristate(ui->toggle_disk_shader_cache,
                                            Settings::values.use_disk_shader_cache,
                                            use_disk_shader_cache);
    ConfigurationShared::SetColoredTristate(ui->toggle_vsync_new, Settings::values.use_vsync_new,
                                            use_vsync_new);
    ConfigurationShared::SetColoredTristate(ui->toggle_async_shaders,
                                            Settings::values.async_shader_compilation,
                                            async_shader_compilation);
    ConfigurationShared::SetColoredTristate(ui->toggle_use_gles, Settings::values.use_gles,
                                            use_gles);
    ConfigurationShared::SetColoredTristate(
        ui->toggle_async_present, Settings::values.async_presentation, async_presentation);
    ConfigurationShared::SetColoredTristate(ui->toggle_skip_slow_draw,
                                            Settings::values.skip_slow_draw, skip_slow_draw);
    ConfigurationShared::SetColoredTristate(ui->toggle_skip_texture_copy,
                                            Settings::values.skip_texture_copy, skip_texture_copy);
    ConfigurationShared::SetColoredTristate(ui->toggle_skip_cpu_write,
                                            Settings::values.skip_cpu_write, skip_cpu_write);
    ConfigurationShared::SetColoredTristate(ui->toggle_upscaling_hack,
                                            Settings::values.upscaling_hack, upscaling_hack);
    ConfigurationShared::SetColoredTristate(ui->spirv_shader_gen, Settings::values.spirv_shader_gen,
                                            spirv_shader_gen);
    ConfigurationShared::SetColoredTristate(ui->toggle_geometry_shader,
                                            Settings::values.geometry_shader, geometry_shader);
    ConfigurationShared::SetColoredTristate(ui->sample_shading, Settings::values.use_sample_shading,
                                            sample_shading);
    ConfigurationShared::SetColoredTristate(ui->toggle_relaxed_precision,
                                            Settings::values.relaxed_precision_decorators,
                                            relaxed_precision_decorators);
    ConfigurationShared::SetColoredTristate(ui->toggle_spirv_validation,
                                            Settings::values.spirv_output_validation,
                                            spirv_output_validation);
    ConfigurationShared::SetColoredTristate(ui->toggle_spirv_legalization,
                                            Settings::values.spirv_output_legalization,
                                            spirv_output_legalization);
}

void ConfigureGraphics::SetPhysicalDeviceComboVisibility(int index) {
    Settings::GraphicsAPI effective_api{};

    // When configuring per-game the physical device combo should be
    // shown either when the global api is used and that is Vulkan or
    // Vulkan is set as the per-game api.
    if (!Settings::IsConfiguringGlobal()) {
        const bool using_global = index == 0;
        if (using_global) {
            effective_api = Settings::values.graphics_api.GetValue(true);
        } else {
            effective_api =
                static_cast<Settings::GraphicsAPI>(index - ConfigurationShared::USE_GLOBAL_OFFSET);
        }
    } else {
        effective_api = static_cast<Settings::GraphicsAPI>(index);
    }

    ui->toggle_use_gles->setVisible(effective_api == Settings::GraphicsAPI::OpenGL);
    ui->physical_device_group->setVisible(effective_api == Settings::GraphicsAPI::Vulkan);
    ui->spirv_shader_gen->setVisible(effective_api == Settings::GraphicsAPI::Vulkan);
    ui->toggle_geometry_shader->setVisible(effective_api == Settings::GraphicsAPI::Vulkan);
    ui->sample_shading->setVisible(effective_api == Settings::GraphicsAPI::Vulkan);
    ui->toggle_relaxed_precision->setVisible(effective_api == Settings::GraphicsAPI::Vulkan);
    ui->toggle_relaxed_precision->setEnabled(Settings::values.spirv_shader_gen.GetValue() == true);
    ui->widget_optimize_spirv->setVisible(effective_api == Settings::GraphicsAPI::Vulkan);
    ui->widget_optimize_spirv->setEnabled(Settings::values.spirv_shader_gen.GetValue() == true);
    ui->toggle_spirv_validation->setEnabled(Settings::values.optimize_spirv_output.GetValue() !=
                                            Settings::OptimizeSpirv::Disabled);
    ui->toggle_spirv_legalization->setEnabled(Settings::values.optimize_spirv_output.GetValue() !=
                                              Settings::OptimizeSpirv::Disabled);
    ui->opengl_renderer_group->setVisible(effective_api == Settings::GraphicsAPI::OpenGL);
}
