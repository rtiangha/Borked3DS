// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.camera

import android.graphics.Bitmap
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.Keep
import androidx.core.graphics.drawable.toBitmap
import coil.executeBlocking
import coil.imageLoader
import coil.request.ImageRequest
import io.github.borked3ds.android.Borked3DSApplication
import io.github.borked3ds.android.NativeLibrary

// Used in native code.
object StillImageCameraHelper {
    private val filePickerLock = Object()
    private var filePickerPath: String? = null

    // Opens file picker for camera.
    @Keep
    @JvmStatic
    fun OpenFilePicker(): String? {
        val emulationActivity = NativeLibrary.sEmulationActivity.get()

        // At this point, we are assuming that we already have permissions as they are
        // needed to launch a game
        emulationActivity?.runOnUiThread {
            val request = PickVisualMediaRequest.Builder()
                .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly).build()
            emulationActivity.openImageLauncher.launch(request)
        } ?: return null

        synchronized(filePickerLock) {
            try {
                filePickerLock.wait()
            } catch (ignored: InterruptedException) {
                // Ignore the interruption and continue
            }
        }
        return filePickerPath
    }

    // Called from EmulationActivity.
    @JvmStatic
    fun OnFilePickerResult(result: String) {
        filePickerPath = result
        synchronized(filePickerLock) { filePickerLock.notifyAll() }
    }

    // Blocking call. Load image from file and crop/resize it to fit in width x height.
    @Keep
    @JvmStatic
    fun LoadImageFromFile(uri: String?, width: Int, height: Int): Bitmap? {
        val context = Borked3DSApplication.appContext
        val request = ImageRequest.Builder(context)
            .data(uri)
            .size(width, height)
            .build()
        return context.imageLoader.executeBlocking(request).drawable?.toBitmap(
            width,
            height,
            Bitmap.Config.ARGB_8888
        )
    }
}
