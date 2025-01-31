// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.features.settings.ui.viewholder

import android.view.View
import androidx.core.content.res.ResourcesCompat
import io.github.borked3ds.android.activities.EmulationActivity
import io.github.borked3ds.android.databinding.ListItemSettingBinding
import io.github.borked3ds.android.features.settings.model.view.RunnableSetting
import io.github.borked3ds.android.features.settings.model.view.SettingsItem
import io.github.borked3ds.android.features.settings.ui.SettingsAdapter

class RunnableViewHolder(
    val binding: ListItemSettingBinding,
    adapter: SettingsAdapter
) : SettingViewHolder(binding.root, adapter) {

    private lateinit var setting: RunnableSetting

    override fun bind(item: SettingsItem) {
        setting = item as RunnableSetting
        if (item.iconId != 0) {
            binding.icon.visibility = View.VISIBLE
            binding.icon.setImageDrawable(
                ResourcesCompat.getDrawable(
                    binding.icon.resources,
                    item.iconId,
                    binding.icon.context.theme
                )
            )
        } else {
            binding.icon.visibility = View.GONE
        }

        binding.textSettingName.setText(item.nameId)
        if (item.descriptionId != 0) {
            binding.textSettingDescription.setText(item.descriptionId)
            binding.textSettingDescription.visibility = View.VISIBLE
        } else {
            binding.textSettingDescription.visibility = View.GONE
        }

        if (setting.value != null) {
            binding.textSettingValue.visibility = View.VISIBLE
            binding.textSettingValue.text = setting.value?.invoke() ?: ""
        } else {
            binding.textSettingValue.visibility = View.GONE
        }

        val textAlpha = if (setting.isEditable) 1f else 0.5f
        binding.textSettingName.alpha = textAlpha
        binding.textSettingDescription.alpha = textAlpha
        binding.textSettingValue.alpha = textAlpha
    }

    override fun onClick(clicked: View) {
        if (!setting.isRuntimeRunnable && EmulationActivity.isRunning()) {
            adapter.onClickDisabledSetting()
        } else {
            setting.runnable.invoke()
        }
    }

    override fun onLongClick(clicked: View): Boolean {
        // no-op
        return true
    }
}
