// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.utils

import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.provider.DocumentsContract
import android.system.Os
import android.util.Pair
import androidx.documentfile.provider.DocumentFile
import io.github.borked3ds.android.Borked3DSApplication
import io.github.borked3ds.android.model.CheapDocument
import okio.ByteString.Companion.readByteString
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.OutputStream
import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

object FileUtil {
    const val PATH_TREE = "tree"
    const val DECODE_METHOD = "UTF-8"
    const val APPLICATION_OCTET_STREAM = "application/octet-stream"
    const val TEXT_PLAIN = "text/plain"

    val context: Context get() = Borked3DSApplication.appContext

    /**
     * Create a file from directory with filename.
     *
     * @param directory parent path for file.
     * @param filename  file display name.
     * @return boolean
     */
    @JvmStatic
    fun createFile(directory: String, filename: String): DocumentFile? {
        return try {
            val directoryUri = Uri.parse(directory)
            val parent = DocumentFile.fromTreeUri(context, directoryUri) ?: return null
            val decodedFilename = URLDecoder.decode(filename, DECODE_METHOD)
            val extensionPosition = decodedFilename.lastIndexOf('.')

            val extension = if (extensionPosition > 0) {
                decodedFilename.substring(extensionPosition)
            } else {
                ""
            }

            val mimeType = if (extension == ".txt") {
                TEXT_PLAIN
            } else {
                APPLICATION_OCTET_STREAM
            }

            val exists = parent.findFile(decodedFilename)
            exists ?: parent.createFile(mimeType, decodedFilename)
        } catch (e: Exception) {
            Log.error("[FileUtil]: Cannot create file, error: ${e.message}")
            null
        }
    }

    /**
     * Create a directory from directory with filename.
     *
     * @param directory     parent path for directory.
     * @param directoryName directory display name.
     * @return boolean
     */
    @JvmStatic
    fun createDir(directory: String, directoryName: String): DocumentFile? {
        return try {
            val directoryUri = Uri.parse(directory)
            val parent = DocumentFile.fromTreeUri(context, directoryUri) ?: return null
            val decodedDirectoryName = URLDecoder.decode(directoryName, DECODE_METHOD)
            val exists = parent.findFile(decodedDirectoryName)
            exists ?: parent.createDirectory(decodedDirectoryName)
        } catch (e: Exception) {
            Log.error("[FileUtil]: Cannot create directory, error: ${e.message}")
            null
        }
    }

    /**
     * Open content uri and return file descriptor to JNI.
     *
     * @param path     Native content uri path
     * @param openMode will be one of "r", "r", "rw", "wa", "rwa"
     * @return file descriptor
     */
    @JvmStatic
    fun openContentUri(path: String, openMode: String): Int {
        return try {
            context.contentResolver.openFileDescriptor(Uri.parse(path), openMode)
                ?.use { parcelFileDescriptor ->
                    parcelFileDescriptor.detachFd()
                } ?: run {
                Log.error("[FileUtil]: Cannot get the file descriptor from uri: $path")
                -1
            }
        } catch (e: Exception) {
            Log.error("[FileUtil]: Cannot open content uri, error: ${e.message}")
            -1
        }
    }

    /**
     * Reference:  https://stackoverflow.com/questions/42186820/documentfile-is-very-slow
     * This function will be faster than DocumentFile.listFiles
     *
     * @param uri     Directory uri.
     * @return CheapDocument lists.
     */
    @JvmStatic
    fun listFiles(uri: Uri): Array<CheapDocument> {
        val columns = arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME,
            DocumentsContract.Document.COLUMN_MIME_TYPE
        )
        var c: Cursor? = null
        val results: MutableList<CheapDocument> = ArrayList()
        try {
            val docId = if (isRootTreeUri(uri)) {
                DocumentsContract.getTreeDocumentId(uri)
            } else {
                DocumentsContract.getDocumentId(uri)
            }

            val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(uri, docId)
            c = context.contentResolver.query(childrenUri, columns, null, null, null)
            while (c?.moveToNext() == true) {
                val documentId = c.getString(0) ?: continue
                val documentName = c.getString(1) ?: continue
                val documentMimeType = c.getString(2) ?: continue
                val documentUri = DocumentsContract.buildDocumentUriUsingTree(uri, documentId)
                val document = CheapDocument(documentName, documentMimeType, documentUri)
                results.add(document)
            }
        } catch (e: Exception) {
            Log.error("[FileUtil]: Cannot list file error: ${e.message}")
        } finally {
            closeQuietly(c)
        }
        return results.toTypedArray()
    }

    /**
     * Check whether given path exists.
     *
     * @param path Native content uri path
     * @return bool
     */
    @JvmStatic
    fun exists(path: String): Boolean {
        var c: Cursor? = null
        return try {
            val uri = Uri.parse(path)
            val columns = arrayOf(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
            c = context.contentResolver.query(uri, columns, null, null, null)
            c?.count ?: 0 > 0
        } catch (e: Exception) {
            Log.info("[FileUtil] Cannot find file from given path, error: ${e.message}")
            false
        } finally {
            closeQuietly(c)
        }
    }

    /**
     * Check whether given path is a directory
     *
     * @param path content uri path
     * @return bool
     */
    @JvmStatic
    fun isDirectory(path: String): Boolean {
        val columns = arrayOf(DocumentsContract.Document.COLUMN_MIME_TYPE)
        var c: Cursor? = null
        return try {
            val uri = Uri.parse(path)
            c = context.contentResolver.query(uri, columns, null, null, null)
            c?.moveToNext() == true && c.getString(0) == DocumentsContract.Document.MIME_TYPE_DIR
        } catch (e: Exception) {
            Log.error("[FileUtil]: Cannot list files, error: ${e.message}")
            false
        } finally {
            closeQuietly(c)
        }
    }

    /**
     * Get file display name from given path
     *
     * @param uri content uri
     * @return String display name
     */
    @JvmStatic
    fun getFilename(uri: Uri): String {
        val columns = arrayOf(DocumentsContract.Document.COLUMN_DISPLAY_NAME)
        var c: Cursor? = null
        return try {
            c = context.contentResolver.query(uri, columns, null, null, null)
            if (c?.moveToNext() == true) {
                c.getString(0) ?: ""
            } else {
                ""
            }
        } catch (e: Exception) {
            Log.error("[FileUtil]: Cannot get file name, error: ${e.message}")
            ""
        } finally {
            closeQuietly(c)
        }
    }

    @JvmStatic
    fun getFilesName(path: String): Array<String?> {
        val uri = Uri.parse(path)
        return listFiles(uri).map { it.filename }.toTypedArray()
    }

    /**
     * Get file size from given path.
     *
     * @param path content uri path
     * @return long file size
     */
    @JvmStatic
    fun getFileSize(path: String): Long {
        val columns = arrayOf(DocumentsContract.Document.COLUMN_SIZE)
        var c: Cursor? = null
        return try {
            val uri = Uri.parse(path)
            c = context.contentResolver.query(uri, columns, null, null, null)
            if (c?.moveToNext() == true) {
                c.getLong(0)
            } else {
                0L
            }
        } catch (e: Exception) {
            Log.error("[FileUtil]: Cannot get file size, error: ${e.message}")
            0L
        } finally {
            closeQuietly(c)
        }
    }

    @JvmStatic
    fun copyFile(
        sourceUri: Uri,
        destinationUri: Uri,
        destinationFilename: String
    ): Boolean {
        return try {
            val destinationParent =
                DocumentFile.fromTreeUri(context, destinationUri) ?: return false
            val filename = URLDecoder.decode(destinationFilename, "UTF-8")
            var destination = destinationParent.findFile(filename)
            if (destination == null) {
                destination = destinationParent.createFile("application/octet-stream", filename)
            }
            if (destination == null) {
                return false
            }

            val input = context.contentResolver.openInputStream(sourceUri) ?: return false
            val output =
                context.contentResolver.openOutputStream(destination.uri, "wt") ?: return false
            val buffer = ByteArray(1024)
            var len: Int
            while (input.read(buffer).also { len = it } != -1) {
                output.write(buffer, 0, len)
            }
            input.close()
            output.flush()
            output.close()
            true
        } catch (e: Exception) {
            Log.error("[FileUtil]: Cannot copy file, error: ${e.message}")
            false
        }
    }

    fun copyUriToInternalStorage(
        sourceUri: Uri?,
        destinationParentPath: String,
        destinationFilename: String
    ): Boolean {
        var input: InputStream? = null
        var output: FileOutputStream? = null
        return try {
            input = context.contentResolver.openInputStream(sourceUri ?: return false)
            output = FileOutputStream("$destinationParentPath/$destinationFilename")
            val buffer = ByteArray(1024)
            var len: Int
            while (input!!.read(buffer).also { len = it } != -1) {
                output.write(buffer, 0, len)
            }
            output.flush()
            true
        } catch (e: Exception) {
            Log.error("[FileUtil]: Cannot copy file, error: ${e.message}")
            false
        } finally {
            input?.closeQuietly()
            output?.closeQuietly()
        }
    }

    fun copyDir(
        sourcePath: String,
        destinationPath: String,
        listener: CopyDirListener?
    ) {
        try {
            val sourceUri = Uri.parse(sourcePath)
            val destinationUri = Uri.parse(destinationPath)
            val files: MutableList<Pair<CheapDocument, DocumentFile>> = ArrayList()
            val dirs: MutableList<Pair<Uri, Uri>> = ArrayList()
            dirs.add(Pair(sourceUri, destinationUri))

            // Searching all files which need to be copied and struct the directory in destination
            while (dirs.isNotEmpty()) {
                val fromDir = DocumentFile.fromTreeUri(context, dirs[0].first)
                val toDir = DocumentFile.fromTreeUri(context, dirs[0].second)
                if (fromDir == null || toDir == null) {
                    continue
                }

                val fromUri = fromDir.uri
                listener?.onSearchProgress(fromUri.path ?: "")
                val documents = listFiles(fromUri)
                for (document in documents) {
                    // Prevent infinite recursion if the source dir is being copied to a dir within itself
                    if (document.filename == toDir.name) {
                        continue
                    }

                    val filename = document.filename
                    if (document.isDirectory) {
                        var target = toDir.findFile(filename)
                        if (target == null || !target.exists()) {
                            target = toDir.createDirectory(filename)
                        }
                        if (target == null) {
                            continue
                        }

                        dirs.add(Pair(document.uri, target.uri))
                    } else {
                        var target = toDir.findFile(filename)
                        if (target == null || !target.exists()) {
                            target = toDir.createFile(document.mimeType, document.filename)
                        }
                        if (target == null) {
                            continue
                        }

                        files.add(Pair(document, target))
                    }
                }
                dirs.removeAt(0)
            }

            var progress = 0
            for (file in files) {
                val to = file.second
                val toUri = to.uri
                val toPath = toUri.path ?: ""
                val toParent = to.parentFile ?: continue
                copyFile(file.first.uri, toParent.uri, to.name ?: continue)
                progress++
                listener?.onCopyProgress(toPath, progress, files.size)
            }
            listener?.onComplete()
        } catch (e: Exception) {
            Log.error("[FileUtil]: Cannot copy directory, error: ${e.message}")
        }
    }

    @JvmStatic
    fun renameFile(path: String, destinationFilename: String): Boolean {
        return try {
            val uri = Uri.parse(path)
            DocumentsContract.renameDocument(
                context.contentResolver,
                uri,
                destinationFilename
            ) != null
        } catch (e: Exception) {
            Log.error("[FileUtil]: Cannot rename file, error: ${e.message}")
            false
        }
    }

    @JvmStatic
    fun deleteDocument(path: String): Boolean {
        return try {
            val uri = Uri.parse(path)
            DocumentsContract.deleteDocument(context.contentResolver, uri)
        } catch (e: Exception) {
            Log.error("[FileUtil]: Cannot delete document, error: ${e.message}")
            false
        }
    }

    @Throws(IOException::class)
    fun getBytesFromFile(file: DocumentFile): ByteArray {
        val uri = file.uri
        val length = getFileSize(uri.toString())

        // You cannot create an array using a long type.
        if (length > Int.MAX_VALUE) {
            // File is too large
            throw IOException("File is too large!")
        }

        val bytes = ByteArray(length.toInt())

        var offset = 0
        var numRead = 0
        context.contentResolver.openInputStream(uri)?.use { inputStream ->
            while (offset < bytes.size &&
                inputStream.read(bytes, offset, bytes.size - offset).also { numRead = it } >= 0
            ) {
                offset += numRead
            }
        }

        // Ensure all the bytes have been read in
        if (offset < bytes.size) {
            throw IOException("Could not completely read file ${file.name}")
        }

        return bytes
    }

    /**
     * Extracts the given zip file into the given directory.
     */
    @Throws(SecurityException::class)
    fun unzipToInternalStorage(zipStream: BufferedInputStream, destDir: File) {
        ZipInputStream(zipStream).use { zis ->
            var entry: ZipEntry? = zis.nextEntry
            while (entry != null) {
                val newFile = File(destDir, entry.name)
                val destinationDirectory = if (entry.isDirectory) newFile else newFile.parentFile

                if (!newFile.canonicalPath.startsWith(destDir.canonicalPath + File.separator)) {
                    throw SecurityException("Zip file attempted path traversal! ${entry.name}")
                }

                if (destinationDirectory?.isDirectory != true && destinationDirectory?.mkdirs() != true) {
                    throw IOException("Failed to create directory $destinationDirectory")
                }

                if (!entry.isDirectory) {
                    newFile.outputStream().use { fos -> zis.copyTo(fos) }
                }
                entry = zis.nextEntry
            }
        }
    }

    fun copyToExternalStorage(
        sourceFile: Uri,
        destinationDir: DocumentFile
    ): DocumentFile? {
        return try {
            val filename = getFilename(sourceFile)
            val destinationFile =
                destinationDir.createFile("application/zip", filename) ?: return null
            destinationFile.outputStream().use { os ->
                sourceFile.inputStream().use { it.copyTo(os) }
            }
            destinationDir.findFile(filename)
        } catch (e: Exception) {
            Log.error("[FileUtil]: Cannot copy file to external storage, error: ${e.message}")
            null
        }
    }

    fun isRootTreeUri(uri: Uri): Boolean {
        val paths = uri.pathSegments
        return paths.size == 2 && PATH_TREE == paths[0]
    }

    @JvmStatic
    fun isNativePath(path: String): Boolean =
        try {
            path[0] == '/'
        } catch (e: StringIndexOutOfBoundsException) {
            Log.error("[FileUtil] Cannot determine the string is native path or not.")
            false
        }

    fun getFreeSpace(context: Context, uri: Uri?): Double =
        try {
            val docTreeUri = DocumentsContract.buildDocumentUriUsingTree(
                uri,
                DocumentsContract.getTreeDocumentId(uri)
            )
            val pfd = context.contentResolver.openFileDescriptor(docTreeUri, "r") ?: return 0.0
            val stats = Os.fstatvfs(pfd.fileDescriptor)
            val spaceInGigaBytes = stats.f_bavail * stats.f_bsize / 1024.0 / 1024 / 1024
            pfd.close()
            spaceInGigaBytes
        } catch (e: Exception) {
            Log.error("[FileUtil] Cannot get storage size.")
            0.0
        }

    fun closeQuietly(closeable: AutoCloseable?) {
        try {
            closeable?.close()
        } catch (rethrown: RuntimeException) {
            throw rethrown
        } catch (ignored: Exception) {
        }
    }

    fun getExtension(uri: Uri): String {
        val fileName = getFilename(uri)
        return fileName.substring(fileName.lastIndexOf(".") + 1)
            .lowercase()
    }

    @Throws(IOException::class)
    fun getStringFromFile(file: File): String =
        String(file.readBytes(), StandardCharsets.UTF_8)

    @Throws(IOException::class)
    fun getStringFromInputStream(stream: InputStream, length: Long = 0L): String =
        if (length == 0L) {
            String(stream.readBytes(), StandardCharsets.UTF_8)
        } else {
            String(stream.readByteString(length.toInt()).toByteArray(), StandardCharsets.UTF_8)
        }

    fun DocumentFile.inputStream(): InputStream =
        context.contentResolver.openInputStream(uri)
            ?: throw IOException("Failed to open input stream")

    fun DocumentFile.outputStream(): OutputStream =
        context.contentResolver.openOutputStream(uri)
            ?: throw IOException("Failed to open output stream")

    fun Uri.inputStream(): InputStream =
        context.contentResolver.openInputStream(this)
            ?: throw IOException("Failed to open input stream")

    fun Uri.outputStream(): OutputStream =
        context.contentResolver.openOutputStream(this)
            ?: throw IOException("Failed to open output stream")

    fun Uri.asDocumentFile(): DocumentFile? =
        DocumentFile.fromSingleUri(context, this)

    interface CopyDirListener {
        fun onSearchProgress(directoryName: String)
        fun onCopyProgress(filename: String, progress: Int, max: Int)
        fun onComplete()
    }
}

private fun InputStream?.closeQuietly() {
    try {
        this?.close()
    } catch (e: IOException) {
        Log.error("[FileUtil]: Cannot close input stream, error: ${e.message}")
    }
}

private fun OutputStream?.closeQuietly() {
    try {
        this?.close()
    } catch (e: IOException) {
        Log.error("[FileUtil]: Cannot close output stream, error: ${e.message}")
    }
}
