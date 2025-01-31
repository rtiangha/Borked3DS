// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.applets

import androidx.annotation.Keep
import io.github.borked3ds.android.NativeLibrary
import io.github.borked3ds.android.fragments.MiiSelectorDialogFragment
import java.io.Serializable

@Keep
object MiiSelector {
    lateinit var data: MiiSelectorData
    val finishLock = Any()

    private fun ExecuteImpl(config: MiiSelectorConfig) {
        val emulationActivity = NativeLibrary.sEmulationActivity.get()
        if (emulationActivity == null) {
            throw IllegalStateException("Emulation activity is not available.")
        }
        data = MiiSelectorData(0, 0)
        val fragment = MiiSelectorDialogFragment.newInstance(config)
        fragment.show(emulationActivity.supportFragmentManager, "mii_selector")
    }

    @JvmStatic
    fun Execute(config: MiiSelectorConfig): MiiSelectorData {
        val emulationActivity = NativeLibrary.sEmulationActivity.get()
        emulationActivity?.runOnUiThread { ExecuteImpl(config) }
            ?: throw IllegalStateException("Emulation activity is not available.")

        synchronized(finishLock) {
            try {
                finishLock.wait()
            } catch (ignored: Exception) {
                // Ignore the interruption and continue
            }
        }
        return data
    }

    @Keep
    class MiiSelectorConfig : Serializable {
        var enableCancelButton = false
        var title: String? = null
        var initiallySelectedMiiIndex: Long = 0

        // List of Miis to display
        lateinit var miiNames: Array<String>
    }

    class MiiSelectorData(var returnCode: Long, var index: Int)
}