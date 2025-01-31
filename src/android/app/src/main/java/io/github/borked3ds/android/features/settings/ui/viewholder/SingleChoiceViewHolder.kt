// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.features.settings.ui.viewholder

import android.view.View
import io.github.borked3ds.android.databinding.ListItemSettingBinding
import io.github.borked3ds.android.features.settings.model.view.SettingsItem
import io.github.borked3ds.android.features.settings.model.view.SingleChoiceSetting
import io.github.borked3ds.android.features.settings.model.view.StringSingleChoiceSetting
import io.github.borked3ds.android.features.settings.ui.SettingsAdapter

class SingleChoiceViewHolder(
    val binding: ListItemSettingBinding,
    adapter: SettingsAdapter
) : SettingViewHolder(binding.root, adapter) {

    private lateinit var setting: SettingsItem

    override fun bind(item: SettingsItem) {
        setting = item
        binding.textSettingName.setText(item.nameId)
        if (item.descriptionId != 0) {
            binding.textSettingDescription.visibility = View.VISIBLE
            binding.textSettingDescription.setText(item.descriptionId)
        } else {
            binding.textSettingDescription.visibility = View.GONE
        }
        binding.textSettingValue.visibility = View.VISIBLE
        binding.textSettingValue.text = getTextSetting()

        val textAlpha = if (setting.isEditable) 1f else 0.5f
        binding.textSettingName.alpha = textAlpha
        binding.textSettingDescription.alpha = textAlpha
        binding.textSettingValue.alpha = textAlpha
    }

    private fun getTextSetting(): String {
        return when (val item = setting) {
            is SingleChoiceSetting -> {
                val resMgr = binding.textSettingDescription.context.resources
                val values = resMgr.getIntArray(item.valuesId)
                values.forEachIndexed { i: Int, value: Int ->
                    if (value == item.selectedValue) {
                        return resMgr.getStringArray(item.choicesId).getOrNull(i) ?: ""
                    }
                }
                ""
            }

            is StringSingleChoiceSetting -> {
                item.values?.forEachIndexed { i: Int, value: String ->
                    if (value == item.selectedValue) {
                        return item.choices.getOrNull(i) ?: ""
                    }
                }
                ""
            }

            else -> ""
        }
    }

    override fun onClick(clicked: View) {
        if (!setting.isEditable) {
            adapter.onClickDisabledSetting()
            return
        }

        when (setting) {
            is SingleChoiceSetting -> {
                adapter.onSingleChoiceClick(
                    setting as SingleChoiceSetting,
                    bindingAdapterPosition
                )
            }

            is StringSingleChoiceSetting -> {
                adapter.onStringSingleChoiceClick(
                    setting as StringSingleChoiceSetting,
                    bindingAdapterPosition
                )
            }
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
