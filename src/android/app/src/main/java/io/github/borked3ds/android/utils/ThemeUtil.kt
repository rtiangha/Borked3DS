// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.utils

import android.content.SharedPreferences
import android.content.res.Configuration
import android.graphics.Color
import android.os.Build
import android.view.View
import androidx.annotation.ColorInt
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.preference.PreferenceManager
import io.github.borked3ds.android.Borked3DSApplication
import io.github.borked3ds.android.R
import io.github.borked3ds.android.features.settings.model.Settings
import io.github.borked3ds.android.ui.main.ThemeProvider
import kotlin.math.roundToInt

object ThemeUtil {
    const val SYSTEM_BAR_ALPHA = 0.9f

    private val preferences: SharedPreferences
        get() =
            PreferenceManager.getDefaultSharedPreferences(Borked3DSApplication.appContext)

    private fun getSelectedStaticThemeColor(): Int {
        val themeIndex = preferences.getInt(Settings.PREF_STATIC_THEME_COLOR, 0)
        val themes = arrayOf(
            R.style.Theme_Borked3DS_Blue,
            R.style.Theme_Borked3DS_Cyan,
            R.style.Theme_Borked3DS_Red,
            R.style.Theme_Borked3DS_Green,
            R.style.Theme_Borked3DS_Lime,
            R.style.Theme_Borked3DS_Yellow,
            R.style.Theme_Borked3DS_Orange,
            R.style.Theme_Borked3DS_Violet,
            R.style.Theme_Borked3DS_Pink,
            R.style.Theme_Borked3DS_Gray
        )
        return themes[themeIndex]
    }

    fun setTheme(activity: AppCompatActivity) {
        setThemeMode(activity)
        if (preferences.getBoolean(Settings.PREF_MATERIAL_YOU, false)) {
            activity.setTheme(R.style.Theme_Borked3DS_Main_MaterialYou)
        } else {
            activity.setTheme(getSelectedStaticThemeColor())
        }

        // Using a specific night mode check because this could apply incorrectly when using the
        // light app mode, dark system mode, and black backgrounds. Launching the settings activity
        // will then show light mode colors/navigation bars but with black backgrounds.
        if (preferences.getBoolean(Settings.PREF_BLACK_BACKGROUNDS, false) &&
            isNightMode(activity)
        ) {
            activity.setTheme(R.style.ThemeOverlay_Borked3DS_Dark)
        }
    }

    fun setThemeMode(activity: AppCompatActivity) {
        val themeMode = getThemeMode(activity)
        activity.delegate.localNightMode = themeMode

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            handleModernSystemBars(activity, themeMode)
        } else {
            handleLegacySystemBars(activity, themeMode)
        }
    }

    @SuppressLint("PrivateApi", "DiscouragedPrivateApi")
    private fun handleModernSystemBars(activity: AppCompatActivity, themeMode: Int) {
        try {
            // Use reflection to avoid direct class references on Android 9
            val window = activity.window
            val decorView = window.decorView
            
            // Equivalent to WindowCompat.getInsetsController()
            val getInsetsControllerMethod = Window::class.java
                .getDeclaredMethod("getInsetsController")
            val controller = getInsetsControllerMethod.invoke(window)

            // Equivalent to controller.isAppearanceLightStatusBars = true
            val lightStatusBarsMethod = controller!!.javaClass
                .getMethod("setSystemBarsAppearance", Int::class.java, Int::class.java)
            
            // Equivalent to controller.isAppearanceLightNavigationBars = true
            val lightNavBarsMethod = controller.javaClass
                .getMethod("setSystemBarsAppearance", Int::class.java, Int::class.java)

            when (themeMode) {
                AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM -> {
                    if (isNightMode(activity)) {
                        lightStatusBarsMethod.invoke(controller, 0, 0x00000008) // Light status bars OFF
                        lightNavBarsMethod.invoke(controller, 0, 0x00000010) // Light nav bars OFF
                    } else {
                        lightStatusBarsMethod.invoke(controller, 0x00000008, 0x00000008)
                        lightNavBarsMethod.invoke(controller, 0x00000010, 0x00000010)
                    }
                }
                AppCompatDelegate.MODE_NIGHT_NO -> {
                    lightStatusBarsMethod.invoke(controller, 0x00000008, 0x00000008)
                    lightNavBarsMethod.invoke(controller, 0x00000010, 0x00000010)
                }
                AppCompatDelegate.MODE_NIGHT_YES -> {
                    lightStatusBarsMethod.invoke(controller, 0, 0x00000008)
                    lightNavBarsMethod.invoke(controller, 0, 0x00000010)
                }
            }
        } catch (e: Exception) {
            // Fallback to legacy method if reflection fails
            handleLegacySystemBars(activity, themeMode)
        }
    }

    @Suppress("DEPRECATION")
    private fun handleLegacySystemBars(activity: AppCompatActivity, themeMode: Int) {
        val nightMode = when (themeMode) {
            AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM -> isNightMode(activity)
            AppCompatDelegate.MODE_NIGHT_YES -> true
            else -> false
        }

        val decorView = activity.window.decorView
        var flags = decorView.systemUiVisibility

        // Status bar (available since API 23)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            flags = if (nightMode) {
                flags and View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR.inv()
            } else {
                flags or View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR
            }
        }

        // Navigation bar (available since API 26)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            flags = if (nightMode) {
                flags and View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR.inv()
            } else {
                flags or View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR
            }
        }

        decorView.systemUiVisibility = flags
    }

    private fun getThemeMode(activity: AppCompatActivity): Int {
        return PreferenceManager.getDefaultSharedPreferences(activity.applicationContext)
            .getInt(Settings.PREF_THEME_MODE, AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM)
    }

    private fun isNightMode(activity: AppCompatActivity): Boolean {
        return (activity.resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) ==
            Configuration.UI_MODE_NIGHT_YES
    }
}

    fun setCorrectTheme(activity: AppCompatActivity) {
        val currentTheme = (activity as ThemeProvider).themeId
        setTheme(activity)
        if (currentTheme != (activity as ThemeProvider).themeId) {
            activity.recreate()
        }
    }

    @ColorInt
    fun getColorWithOpacity(@ColorInt color: Int, alphaFactor: Float): Int {
        return Color.argb(
            (alphaFactor * Color.alpha(color)).roundToInt(),
            Color.red(color),
            Color.green(color),
            Color.blue(color)
        )
    }

    // Listener that detects if the theme keys are being changed from the setting menu and recreates the activity
    private var listener: SharedPreferences.OnSharedPreferenceChangeListener? = null

    fun ThemeChangeListener(activity: AppCompatActivity) {
        listener = SharedPreferences.OnSharedPreferenceChangeListener { _, key ->
            val relevantKeys = listOf(
                Settings.PREF_STATIC_THEME_COLOR,
                Settings.PREF_MATERIAL_YOU,
                Settings.PREF_BLACK_BACKGROUNDS
            )
            if (key in relevantKeys) {
                activity.recreate()
            }
        }
        preferences.registerOnSharedPreferenceChangeListener(listener)
    }
}

@RequiresApi(Build.VERSION_CODES.R)
private object Api30PlusSystemBars {
    fun configure(activity: AppCompatActivity, themeMode: Int) {
        val windowController = WindowCompat.getInsetsController(
            activity.window,
            activity.window.decorView
        )
        when (themeMode) {
            AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM -> {
                if (ThemeUtil.isNightMode(activity)) {
                    windowController.isAppearanceLightStatusBars = false
                    windowController.isAppearanceLightNavigationBars = false
                } else {
                    windowController.isAppearanceLightStatusBars = true
                    windowController.isAppearanceLightNavigationBars = true
                }
            }
            AppCompatDelegate.MODE_NIGHT_NO -> {
                windowController.isAppearanceLightStatusBars = true
                windowController.isAppearanceLightNavigationBars = true
            }
            AppCompatDelegate.MODE_NIGHT_YES -> {
                windowController.isAppearanceLightStatusBars = false
                windowController.isAppearanceLightNavigationBars = false
            }
        }
    }
}
