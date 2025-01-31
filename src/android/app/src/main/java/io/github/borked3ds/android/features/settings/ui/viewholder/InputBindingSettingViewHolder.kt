// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.features.settings.ui.viewholder

import android.view.View
import androidx.preference.PreferenceManager
import io.github.borked3ds.android.Borked3DSApplication
import io.github.borked3ds.android.databinding.ListItemSettingBinding
import io.github.borked3ds.android.features.settings.model.view.InputBindingSetting
import io.github.borked3ds.android.features.settings.model.view.SettingsItem
import io.github.borked3ds.android.features.settings.ui.SettingsAdapter

class InputBindingSettingViewHolder(
    val binding: ListItemSettingBinding,
    adapter: SettingsAdapter
) : SettingViewHolder(binding.root, adapter) {

    private lateinit var setting: InputBindingSetting

    override fun bind(item: SettingsItem) {
        val preferences =
            PreferenceManager.getDefaultSharedPreferences(Borked3DSApplication.appContext)
        setting = item as InputBindingSetting
        binding.textSettingName.setText(item.nameId)
        val uiString = preferences.getString(setting.abstractSetting.key, "") ?: ""
        if (uiString.isNotEmpty()) {
            binding.textSettingDescription.visibility = View.GONE
            binding.textSettingValue.visibility = View.VISIBLE
            binding.textSettingValue.text = uiString
        } else {
            binding.textSettingDescription.visibility = View.GONE
            binding.textSettingValue.visibility = View.GONE
        }

        val textAlpha = if (setting.isEditable) 1f else 0.5f
        binding.textSettingName.alpha = textAlpha
        binding.textSettingDescription.alpha = textAlpha
        binding.textSettingValue.alpha = textAlpha
    }

    override fun onClick(clicked: View) {
        if (setting.isEditable) {
            adapter.onInputBindingClick(setting, bindingAdapterPosition)
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
