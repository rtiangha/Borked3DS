// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.utils

import android.content.Context
import android.content.Intent
import androidx.documentfile.provider.DocumentFile

object FileBrowserHelper {
    fun getSelectedFiles(
        result: Intent,
        context: Context,
        extension: List<String?>
    ): Array<String>? {
        val clipData = result.clipData
        val files: MutableList<DocumentFile?> = ArrayList()
        if (clipData == null) {
            result.data?.let { uri ->
                files.add(DocumentFile.fromSingleUri(context, uri))
            }
        } else {
            for (i in 0 until clipData.itemCount) {
                val item = clipData.getItemAt(i)
                item.uri?.let { uri ->
                    files.add(DocumentFile.fromSingleUri(context, uri))
                }
            }
        }
        if (files.isNotEmpty()) {
            val filePaths: MutableList<String> = ArrayList()
            for (file in files) {
                val filename = file?.name
                val extensionStart = filename?.lastIndexOf('.') ?: 0
                if (extensionStart > 0) {
                    val fileExtension = filename?.substring(extensionStart + 1)
                    if (extension.contains(fileExtension)) {
                        file?.uri?.let { uri ->
                            filePaths.add(uri.toString())
                        }
                    }
                }
            }
            return if (filePaths.isEmpty()) null else filePaths.toTypedArray()
        }
        return null
    }
}