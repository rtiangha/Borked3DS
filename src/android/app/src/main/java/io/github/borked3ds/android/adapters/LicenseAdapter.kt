// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.adapters

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.RecyclerView
import androidx.recyclerview.widget.RecyclerView.ViewHolder
import io.github.borked3ds.android.Borked3DSApplication
import io.github.borked3ds.android.databinding.ListItemSettingBinding
import io.github.borked3ds.android.fragments.LicenseBottomSheetDialogFragment
import io.github.borked3ds.android.model.License

class LicenseAdapter(private val activity: AppCompatActivity, var licenses: List<License>) :
    RecyclerView.Adapter<LicenseAdapter.LicenseViewHolder>(),
    View.OnClickListener {
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): LicenseViewHolder {
        val binding =
            ListItemSettingBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        binding.root.setOnClickListener(this)
        return LicenseViewHolder(binding)
    }

    override fun getItemCount(): Int = licenses.size

    override fun onBindViewHolder(holder: LicenseViewHolder, position: Int) {
        holder.bind(licenses[position])
    }

    override fun onClick(view: View) {
        val license = (view.tag as LicenseViewHolder).license
        LicenseBottomSheetDialogFragment.newInstance(license)
            .show(activity.supportFragmentManager, LicenseBottomSheetDialogFragment.TAG)
    }

    inner class LicenseViewHolder(val binding: ListItemSettingBinding) : ViewHolder(binding.root) {
        lateinit var license: License

        init {
            itemView.tag = this
        }

        fun bind(license: License) {
            this.license = license

            val context = Borked3DSApplication.appContext
            binding.textSettingName.text = context.getString(license.titleId)
            binding.textSettingDescription.text = context.getString(license.descriptionId)
        }
    }
}
