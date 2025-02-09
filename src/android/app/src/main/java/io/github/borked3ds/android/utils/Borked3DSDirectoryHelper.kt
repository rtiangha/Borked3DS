// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.utils

import android.content.Intent
import android.net.Uri
import androidx.fragment.app.FragmentActivity
import androidx.lifecycle.ViewModelProvider
import io.github.borked3ds.android.fragments.Borked3DSDirectoryDialogFragment
import io.github.borked3ds.android.fragments.CopyDirProgressDialog
import io.github.borked3ds.android.model.SetupCallback
import io.github.borked3ds.android.viewmodel.HomeViewModel

/**
 * Borked3DS directory initialization ui flow controller.
 */
class Borked3DSDirectoryHelper(private val fragmentActivity: FragmentActivity) {
    fun showBorked3DSDirectoryDialog(result: Uri, callback: SetupCallback? = null) {
        val borked3dsDirectoryDialog = Borked3DSDirectoryDialogFragment.newInstance(
            fragmentActivity,
            result.toString(),
            Borked3DSDirectoryDialogFragment.Listener { moveData: Boolean, path: Uri ->
                val previous = PermissionsHandler.borked3dsDirectory
                // Do noting if user select the previous path.
                if (path == previous) {
                    return@Listener
                }

                val takeFlags = Intent.FLAG_GRANT_WRITE_URI_PERMISSION or
                        Intent.FLAG_GRANT_READ_URI_PERMISSION
                fragmentActivity.contentResolver.takePersistableUriPermission(
                    path,
                    takeFlags
                )
                if (!moveData || previous.toString().isEmpty()) {
                    initializeBorked3DSDirectory(path)
                    callback?.onStepCompleted()
                    val viewModel = ViewModelProvider(fragmentActivity)[HomeViewModel::class.java]
                    viewModel.setUserDir(fragmentActivity, path.path!!)
                    viewModel.setPickingUserDir(false)
                    return@Listener
                }

                // If user check move data, show copy progress dialog.
                CopyDirProgressDialog.newInstance(fragmentActivity, previous, path, callback)
                    ?.show(fragmentActivity.supportFragmentManager, CopyDirProgressDialog.TAG)
            })
        borked3dsDirectoryDialog.show(
            fragmentActivity.supportFragmentManager,
            Borked3DSDirectoryDialogFragment.TAG
        )
    }

    companion object {
        fun initializeBorked3DSDirectory(path: Uri) {
            PermissionsHandler.setBorked3DSDirectory(path.toString())
            DirectoryInitialization.resetBorked3DSDirectoryState()
            DirectoryInitialization.start()
        }
    }
}
