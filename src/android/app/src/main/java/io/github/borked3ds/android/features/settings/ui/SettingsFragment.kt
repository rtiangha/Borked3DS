// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.features.settings.ui

import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import io.github.borked3ds.android.databinding.FragmentSettingsBinding
import io.github.borked3ds.android.features.settings.model.AbstractSetting
import io.github.borked3ds.android.features.settings.model.view.SettingsItem
import io.github.borked3ds.android.utils.ViewUtils.updateMargins

class SettingsFragment : Fragment(), SettingsFragmentView {
    override var activityView: SettingsActivityView? = null

    private val fragmentPresenter = SettingsFragmentPresenter(this)
    private var settingsAdapter: SettingsAdapter? = null

    private var _binding: FragmentSettingsBinding? = null
    private val binding get() = _binding!!

    override fun onAttach(context: Context) {
        super.onAttach(context)
        activityView = requireActivity() as? SettingsActivityView
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val menuTag = requireArguments().getString(ARGUMENT_MENU_TAG)
        val gameId = requireArguments().getString(ARGUMENT_GAME_ID)
        if (menuTag == null || gameId == null) {
            throw IllegalStateException("menuTag and gameId must not be null")
        }
        fragmentPresenter.onCreate(menuTag, gameId)
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSettingsBinding.inflate(layoutInflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        settingsAdapter = SettingsAdapter(this, requireActivity())
        binding.listSettings.apply {
            adapter = settingsAdapter
            layoutManager = LinearLayoutManager(activity)
        }
        settingsAdapter?.let {
            fragmentPresenter.onViewCreated(it)
        }

        setInsets()
    }

    override fun onDetach() {
        super.onDetach()
        activityView = null
        settingsAdapter?.closeDialog()
    }

    override fun showSettingsList(settingsList: ArrayList<SettingsItem>) {
        settingsAdapter?.setSettingsList(settingsList)
    }

    override fun loadSettingsList() {
        fragmentPresenter.loadSettingsList()
    }

    override fun loadSubMenu(menuKey: String) {
        val gameId = requireArguments().getString(ARGUMENT_GAME_ID)
        if (gameId != null) {
            activityView?.showSettingsFragment(menuKey, true, gameId)
        }
    }

    override fun showToastMessage(message: String?, isLong: Boolean) {
        activityView?.showToastMessage(message ?: "", isLong)
    }

    override fun putSetting(setting: AbstractSetting) {
        fragmentPresenter.putSetting(setting)
    }

    override fun onSettingChanged() {
        activityView?.onSettingChanged()
    }

    private fun setInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(
            binding.root
        ) { _: View, windowInsets: WindowInsetsCompat ->
            val barInsets = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
            val cutoutInsets = windowInsets.getInsets(WindowInsetsCompat.Type.displayCutout())

            val leftInsets = barInsets.left + cutoutInsets.left
            val rightInsets = barInsets.right + cutoutInsets.right

            binding.listSettings.updateMargins(left = leftInsets, right = rightInsets)
            binding.listSettings.updatePadding(bottom = barInsets.bottom)
            windowInsets
        }
    }

    companion object {
        private const val ARGUMENT_MENU_TAG = "menu_tag"
        private const val ARGUMENT_GAME_ID = "game_id"

        fun newInstance(menuTag: String?, gameId: String?): Fragment {
            val fragment = SettingsFragment()
            val arguments = Bundle()
            arguments.putString(ARGUMENT_MENU_TAG, menuTag)
            arguments.putString(ARGUMENT_GAME_ID, gameId)
            fragment.arguments = arguments
            return fragment
        }
    }
}