// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.model

import android.content.Intent
import android.net.Uri
import android.os.Parcelable
import io.github.borked3ds.android.Borked3DSApplication
import io.github.borked3ds.android.activities.EmulationActivity
import kotlinx.parcelize.Parcelize
import kotlinx.serialization.Serializable
import java.util.HashSet

@Parcelize
@Serializable
class Game(
    val title: String = "",
    val description: String = "",
    val path: String = "",
    val titleId: Long = 0L,
    val company: String = "",
    val regions: String = "",
    val isInstalled: Boolean = false,
    val isSystemTitle: Boolean = false,
    val isVisibleSystemTitle: Boolean = false,
    val icon: IntArray? = null,
    val filename: String
) : Parcelable {
    val keyAddedToLibraryTime get() = "${filename}_AddedToLibraryTime"
    val keyLastPlayedTime get() = "${filename}_LastPlayed"

    val launchIntent: Intent
        get() = Intent(Borked3DSApplication.appContext, EmulationActivity::class.java).apply {
            action = Intent.ACTION_VIEW
            data = if (isInstalled) {
                Borked3DSApplication.documentsTree.getUri(path)
            } else {
                Uri.parse(path)
            }
        }

    override fun equals(other: Any?): Boolean {
        if (other !is Game) {
            return false
        }

        return hashCode() == other.hashCode()
    }

    override fun hashCode(): Int {
        var result = title.hashCode()
        result = 31 * result + description.hashCode()
        result = 31 * result + regions.hashCode()
        result = 31 * result + path.hashCode()
        result = 31 * result + titleId.hashCode()
        result = 31 * result + company.hashCode()
        return result
    }

    companion object {
        val allExtensions: Set<String> get() = extensions + badExtensions

        val extensions: Set<String> = HashSet(
            listOf("3ds", "3dsx", "elf", "axf", "cci", "cxi", "app")
        )

        val badExtensions: Set<String> = HashSet(
            listOf("rar", "zip", "7z", "torrent", "tar", "gz")
        )
    }
}
