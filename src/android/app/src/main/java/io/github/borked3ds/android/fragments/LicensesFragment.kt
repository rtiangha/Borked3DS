// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.ViewGroup.MarginLayoutParams
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.navigation.findNavController
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.android.material.transition.MaterialSharedAxis
import io.github.borked3ds.android.R
import io.github.borked3ds.android.adapters.LicenseAdapter
import io.github.borked3ds.android.databinding.FragmentLicensesBinding
import io.github.borked3ds.android.model.License
import io.github.borked3ds.android.viewmodel.HomeViewModel

class LicensesFragment : Fragment() {
    private var _binding: FragmentLicensesBinding? = null
    private val binding get() = _binding!!

    private val homeViewModel: HomeViewModel by activityViewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enterTransition = MaterialSharedAxis(MaterialSharedAxis.X, true)
        returnTransition = MaterialSharedAxis(MaterialSharedAxis.X, false)
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentLicensesBinding.inflate(layoutInflater)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        homeViewModel.setNavigationVisibility(visible = false, animated = true)
        homeViewModel.setStatusBarShadeVisibility(visible = false)

        binding.toolbarLicenses.setNavigationOnClickListener {
            binding.root.findNavController().popBackStack()
        }

        val licenses = listOf(
            License(
                R.string.license_adreno_tools,
                R.string.license_adreno_tools_description,
                R.string.license_adreno_tools_link,
                R.string.license_adreno_tools_copyright,
                R.string.license_adreno_tools_text
            ),
            License(
                titleId = R.string.license_boost,
                descriptionId = R.string.license_boost_description,
                linkId = R.string.license_boost_link,
                licenseId = R.string.license_boost_text
            ),
            License(
                R.string.license_borked3ds,
                R.string.license_borked3ds_description,
                R.string.license_borked3ds_link,
                R.string.license_borked3ds_copyright,
                R.string.license_gpl2_text
            ),
            License(
                R.string.license_catch2,
                R.string.license_catch2_description,
                R.string.license_catch2_link,
                R.string.license_catch2_copyright,
                R.string.license_boost_text
            ),
            License(
                R.string.license_cryptopp,
                R.string.license_cryptopp_description,
                R.string.license_cryptopp_link,
                R.string.license_cryptopp_copyright,
                R.string.license_cryptopp_text
            ),
            License(
                R.string.license_cubeb,
                R.string.license_cubeb_description,
                R.string.license_cubeb_link,
                R.string.license_cubeb_copyright,
                R.string.license_isc_text
            ),
            License(
                R.string.license_dynarmic,
                R.string.license_dynarmic_description,
                R.string.license_dynarmic_link,
                R.string.license_dynarmic_copyright,
                R.string.license_0bsd_text
            ),
            License(
                R.string.license_enet,
                R.string.license_enet_description,
                R.string.license_enet_link,
                R.string.license_enet_copyright,
                R.string.license_mit
            ),
            License(
                R.string.license_faad2,
                R.string.license_faad2_description,
                R.string.license_faad2_link,
                R.string.license_faad2_copyright,
                R.string.license_gpl2_text
            ),
            License(
                R.string.license_ffmpeg,
                R.string.license_ffmpeg_description,
                R.string.license_ffmpeg_link,
                R.string.license_ffmpeg_copyright,
                R.string.license_ffmpeg_text
            ),
            License(
                R.string.license_fmt,
                R.string.license_fmt_description,
                R.string.license_fmt_link,
                R.string.license_fmt_copyright,
                R.string.license_mit
            ),
            License(
                R.string.license_glad,
                R.string.license_glad_description,
                R.string.license_glad_link,
                R.string.license_glad_copyright,
                R.string.license_mit
            ),
            License(
                titleId = R.string.license_glslang,
                descriptionId = R.string.license_glslang_description,
                linkId = R.string.license_glslang_link,
                licenseLinkId = R.string.license_glslang_link_license
            ),
            License(
                R.string.license_httplib,
                R.string.license_httplib_description,
                R.string.license_httplib_link,
                R.string.license_httplib_copyright,
                R.string.license_mit
            ),
            License(
                R.string.license_inih,
                R.string.license_inih_description,
                R.string.license_inih_link,
                R.string.license_inih_copyright,
                R.string.license_newbsd_text
            ),
            License(
                R.string.license_libressl,
                R.string.license_libressl_description,
                R.string.license_libressl_link,
                R.string.license_libressl_copyright,
                R.string.license_libressl_license_text
            ),
            License(
                R.string.license_libusb,
                R.string.license_libusb_description,
                R.string.license_libusb_link,
                R.string.license_libusb_copyright,
                R.string.license_lgpl21_text
            ),
            License(
                R.string.license_libyuv,
                R.string.license_libyuv_description,
                R.string.license_libyuv_link,
                R.string.license_libyuv_copyright,
                R.string.license_bsd3_text
            ),
            License(
                R.string.license_lodepng,
                R.string.license_lodepng_description,
                R.string.license_lodepng_link,
                R.string.license_lodepng_copyright,
                R.string.license_zlib_text
            ),
            License(
                R.string.license_nihstro,
                R.string.license_nihstro_description,
                R.string.license_nihstro_link,
                R.string.license_nihstro_copyright,
                R.string.license_bsd3_text
            ),
            License(
                R.string.license_oaknut,
                R.string.license_oaknut_description,
                R.string.license_oaknut_link,
                R.string.license_oaknut_copyright,
                R.string.license_mit
            ),
            License(
                R.string.license_oboe,
                R.string.license_oboe_description,
                R.string.license_oboe_link,
                R.string.license_oboe_copyright,
                R.string.license_apache2_text
            ),
            License(
                R.string.license_openal,
                R.string.license_openal_description,
                R.string.license_openal_link,
                R.string.license_openal_copyright,
                R.string.license_openal_text
            ),
            License(
                R.string.license_sdl,
                R.string.license_sdl_description,
                R.string.license_sdl_link,
                R.string.license_sdl_copyright,
                R.string.license_zlib_text
            ),
            License(
                R.string.license_sirit,
                R.string.license_sirit_description,
                R.string.license_sirit_link,
                R.string.license_sirit_copyright,
                R.string.license_bsd3_text
            ),
            License(
                R.string.license_soundtouch,
                R.string.license_soundtouch_description,
                R.string.license_soundtouch_link,
                R.string.license_soundtouch_copyright,
                R.string.license_lgpl21_text
            ),
            License(
                R.string.license_spirvtools,
                R.string.license_spirvtools_description,
                R.string.license_spirvtools_link,
                R.string.license_spirvtools_copyright,
                R.string.license_apache2_text
            ),
            License(
                R.string.license_teakra,
                R.string.license_teakra_description,
                R.string.license_teakra_link,
                R.string.license_teakra_copyright,
                R.string.license_mit
            ),
            License(
                R.string.license_tracy,
                R.string.license_tracy_description,
                R.string.license_tracy_link,
                R.string.license_tracy_copyright,
                R.string.license_bsd3_text
            ),
            License(
                R.string.license_vma,
                R.string.license_vma_description,
                R.string.license_vma_link,
                R.string.license_vma_copyright,
                R.string.license_mit
            ),
            License(
                R.string.license_zstd,
                R.string.license_zstd_description,
                R.string.license_zstd_link,
                R.string.license_zstd_copyright,
                R.string.license_zstd_text
            )
        )

        binding.listLicenses.apply {
            layoutManager = LinearLayoutManager(requireContext())
            adapter = LicenseAdapter(requireActivity() as AppCompatActivity, licenses)
        }

        setInsets()
    }

    private fun setInsets() =
        ViewCompat.setOnApplyWindowInsetsListener(
            binding.root
        ) { _: View, windowInsets: WindowInsetsCompat ->
            val barInsets = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
            val cutoutInsets = windowInsets.getInsets(WindowInsetsCompat.Type.displayCutout())

            val leftInsets = barInsets.left + cutoutInsets.left
            val rightInsets = barInsets.right + cutoutInsets.right

            val mlpAppBar = binding.toolbarLicenses.layoutParams as MarginLayoutParams
            mlpAppBar.leftMargin = leftInsets
            mlpAppBar.rightMargin = rightInsets
            binding.toolbarLicenses.layoutParams = mlpAppBar

            val mlpScrollAbout = binding.listLicenses.layoutParams as MarginLayoutParams
            mlpScrollAbout.leftMargin = leftInsets
            mlpScrollAbout.rightMargin = rightInsets
            binding.listLicenses.layoutParams = mlpScrollAbout

            binding.listLicenses.updatePadding(bottom = barInsets.bottom)

            windowInsets
        }
}
