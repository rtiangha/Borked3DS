// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.features.settings.ui.viewholder

import android.view.View
import io.github.borked3ds.android.databinding.ListItemSettingBinding
import io.github.borked3ds.android.features.settings.model.AbstractFloatSetting
import io.github.borked3ds.android.features.settings.model.AbstractIntSetting
import io.github.borked3ds.android.features.settings.model.FloatSetting
import io.github.borked3ds.android.features.settings.model.ScaledFloatSetting
import io.github.borked3ds.android.features.settings.model.view.SettingsItem
import io.github.borked3ds.android.features.settings.model.view.SliderSetting
import io.github.borked3ds.android.features.settings.ui.SettingsAdapter

class SliderViewHolder(
    val binding: ListItemSettingBinding,
    adapter: SettingsAdapter
) : SettingViewHolder(binding.root, adapter) {

    private lateinit var setting: SliderSetting

    override fun bind(item: SettingsItem) {
        setting = item as SliderSetting
        binding.textSettingName.setText(item.nameId)
        if (item.descriptionId != 0) {
            binding.textSettingDescription.visibility = View.VISIBLE
            binding.textSettingDescription.setText(item.descriptionId)
        } else {
            binding.textSettingDescription.visibility = View.GONE
        }
        binding.textSettingValue.visibility = View.VISIBLE
        binding.textSettingValue.text = when (val settingValue = setting.setting) {
            is ScaledFloatSetting -> "${settingValue.float.toInt()}${setting.units}"
            is FloatSetting -> "${settingValue.float}${setting.units}"
            is AbstractIntSetting -> "${settingValue.int}${setting.units}"
            else -> ""
        }

        val textAlpha = if (setting.isEditable) 1f else 0.5f
        binding.textSettingName.alpha = textAlpha
        binding.textSettingDescription.alpha = textAlpha
        binding.textSettingValue.alpha = textAlpha
    }

    override fun onClick(clicked: View) {
        if (setting.isEditable) {
            adapter.onSliderClick(setting, bindingAdapterPosition)
        } else {
            adapter.onClickDisabledSetting()
        }
    }

    override fun onLongClick(clicked: View): Boolean {
        if (setting.isEditable) {
            return adapter.onLongClick(setting.setting!!, bindingAdapterPosition)
        } else {
            adapter.onClickDisabledSetting()
        }
        return false
    }
}
