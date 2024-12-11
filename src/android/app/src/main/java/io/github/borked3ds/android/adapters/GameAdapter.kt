// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.adapters

import android.content.Context
import android.content.Intent
import android.content.pm.ShortcutInfo
import android.content.pm.ShortcutManager
import android.graphics.drawable.BitmapDrawable
import android.graphics.drawable.Icon
import android.net.Uri
import android.os.SystemClock
import android.text.TextUtils
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.PopupMenu
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.ViewModelProvider
import androidx.navigation.findNavController
import androidx.preference.PreferenceManager
import androidx.recyclerview.widget.AsyncDifferConfig
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.button.MaterialButton
import com.google.android.material.color.MaterialColors
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import io.github.borked3ds.android.Borked3DSApplication
import io.github.borked3ds.android.HomeNavigationDirections
import io.github.borked3ds.android.NativeLibrary
import io.github.borked3ds.android.R
import io.github.borked3ds.android.adapters.GameAdapter.GameViewHolder
import io.github.borked3ds.android.databinding.CardGameBinding
import io.github.borked3ds.android.features.cheats.ui.CheatsFragmentDirections
import io.github.borked3ds.android.fragments.IndeterminateProgressDialogFragment
import io.github.borked3ds.android.model.FileUtil
import io.github.borked3ds.android.model.Game
import io.github.borked3ds.android.utils.GameIconUtils
import io.github.borked3ds.android.utils.DirectoryInitialization.userDirectory
import io.github.borked3ds.android.viewmodel.GamesViewModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class GameAdapter(private val activity: AppCompatActivity, private val inflater: LayoutInflater) :
    ListAdapter<Game, GameViewHolder>(AsyncDifferConfig.Builder(DiffCallback()).build()),
    View.OnClickListener, View.OnLongClickListener {
    private var lastClickTime = 0L

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): GameViewHolder {
        // Create a new view.
        val binding = CardGameBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        binding.cardGame.setOnClickListener(this)
        binding.cardGame.setOnLongClickListener(this)

        // Use that view to create a ViewHolder.
        return GameViewHolder(binding)
    }

    override fun onBindViewHolder(holder: GameViewHolder, position: Int) {
        holder.bind(currentList[position])
    }

    override fun getItemCount(): Int = currentList.size

    /**
     * Launches the game that was clicked on.
     *
     * @param view The card representing the game the user wants to play.
     */
    override fun onClick(view: View) {
        // Double-click prevention, using threshold of 1000 ms
        if (SystemClock.elapsedRealtime() - lastClickTime < 1000) {
            return
        }
        lastClickTime = SystemClock.elapsedRealtime()

        val holder = view.tag as GameViewHolder
        gameExists(holder)

        val preferences =
            PreferenceManager.getDefaultSharedPreferences(Borked3DSApplication.appContext)
        preferences.edit()
            .putLong(
                holder.game.keyLastPlayedTime,
                System.currentTimeMillis()
            )
            .apply()

        val action = HomeNavigationDirections.actionGlobalEmulationActivity(holder.game)
        view.findNavController().navigate(action)
    }

    /**
     * Opens the about game dialog for the game that was clicked on.
     *
     * @param view The view representing the game the user wants to play.
     */
    override fun onLongClick(view: View): Boolean {
        val context = view.context
        val holder = view.tag as GameViewHolder
        gameExists(holder)

        if (holder.game.titleId == 0L) {
            MaterialAlertDialogBuilder(context)
                .setTitle(R.string.properties)
                .setMessage(R.string.properties_not_loaded)
                .setPositiveButton(android.R.string.ok, null)
                .show()
        } else {
            showAboutGameDialog(context, holder.game, holder, view)
        }
        return true
    }

    // Triggers a library refresh if the user clicks on stale data
    private fun gameExists(holder: GameViewHolder): Boolean {
        if (holder.game.isInstalled) {
            return true
        }

        val gameExists = DocumentFile.fromSingleUri(
            Borked3DSApplication.appContext,
            Uri.parse(holder.game.path)
        )?.exists() == true
        return if (!gameExists) {
            Toast.makeText(
                Borked3DSApplication.appContext,
                R.string.loader_error_file_not_found,
                Toast.LENGTH_LONG
            ).show()

            ViewModelProvider(activity)[GamesViewModel::class.java].reloadGames(true)
            false
        } else {
            true
        }
    }

    inner class GameViewHolder(val binding: CardGameBinding) :
        RecyclerView.ViewHolder(binding.root) {
        lateinit var game: Game

        init {
            binding.cardGame.tag = this
        }

        fun bind(game: Game) {
            this.game = game

            binding.imageGameScreen.scaleType = ImageView.ScaleType.CENTER_CROP
            GameIconUtils.loadGameIcon(activity, game, binding.imageGameScreen)

            binding.textGameTitle.visibility = if (game.title.isEmpty()) {
                View.GONE
            } else {
                View.VISIBLE
            }
            binding.textCompany.visibility = if (game.company.isEmpty()) {
                View.GONE
            } else {
                View.VISIBLE
            }
            binding.textGameId.visibility = if (game.titleId == 0L) {
                View.GONE
            } else {
                View.VISIBLE
            }

            binding.textGameTitle.text = game.title
            binding.textCompany.text = game.company
            binding.textGameRegion.text = game.regions
            binding.textGameId.text = String.format("ID: %016X", game.titleId)
            binding.textFilename.text = game.filename

            val backgroundColorId =
                if (
                    isValidGame(
                        game.filename.substring(game.filename.lastIndexOf(".") + 1).lowercase()
                    )
                ) {
                    com.google.android.material.R.attr.colorSurface
                } else {
                    com.google.android.material.R.attr.colorErrorContainer
                }
            binding.cardContents.setBackgroundColor(
                MaterialColors.getColor(
                    binding.cardContents,
                    backgroundColorId
                )
            )

            binding.textGameTitle.postDelayed(
                {
                    binding.textGameTitle.ellipsize = TextUtils.TruncateAt.MARQUEE
                    binding.textGameTitle.isSelected = true

                    binding.textCompany.ellipsize = TextUtils.TruncateAt.MARQUEE
                    binding.textCompany.isSelected = true

                    binding.textGameRegion.ellipsize = TextUtils.TruncateAt.MARQUEE
                    binding.textGameRegion.isSelected = true
                    binding.textGameId.ellipsize = TextUtils.TruncateAt.MARQUEE
                    binding.textGameId.isSelected = true

                    binding.textFilename.ellipsize = TextUtils.TruncateAt.MARQUEE
                    binding.textFilename.isSelected = true
                },
                3000
            )
        }
    }

    private fun showAboutGameDialog(
        context: Context,
        game: Game,
        holder: GameViewHolder,
        view: View
    ) {
        val bottomSheetView = inflater.inflate(R.layout.dialog_about_game, null)

        val game_id = String.format("%016X", game.titleId)
        val game_filename = game.filename
        val id_label = context.getString(R.string.id_label)
        val file_label = context.getString(R.string.file_label)

        val bottomSheetDialog = BottomSheetDialog(context)
        bottomSheetDialog.setContentView(bottomSheetView)

        bottomSheetView.findViewById<TextView>(R.id.about_game_title).text = game.title
        bottomSheetView.findViewById<TextView>(R.id.about_game_company).text = game.company
        bottomSheetView.findViewById<TextView>(R.id.about_game_region).text = game.regions
        bottomSheetView.findViewById<TextView>(R.id.about_game_id).text =
            "$id_label: $game_id"
        bottomSheetView.findViewById<TextView>(R.id.about_game_filename).text =
            "$file_label: $game_filename"
        GameIconUtils.loadGameIcon(activity, game, bottomSheetView.findViewById(R.id.game_icon))

        bottomSheetView.findViewById<MaterialButton>(R.id.about_game_play).setOnClickListener {
            val action = HomeNavigationDirections.actionGlobalEmulationActivity(holder.game)
            view.findNavController().navigate(action)
        }

        bottomSheetView.findViewById<MaterialButton>(R.id.game_shortcut).setOnClickListener {
            val shortcutManager = activity.getSystemService(ShortcutManager::class.java)

            CoroutineScope(Dispatchers.IO).launch {
                val bitmap =
                    (bottomSheetView.findViewById<ImageView>(R.id.game_icon).drawable as BitmapDrawable).bitmap
                val icon = Icon.createWithBitmap(bitmap)

                val shortcut = ShortcutInfo.Builder(context, game.title)
                    .setShortLabel(game.title)
                    .setIcon(icon)
                    .setIntent(game.launchIntent.apply {
                        putExtra("launched_from_shortcut", true)
                    })
                    .build()
                shortcutManager.requestPinShortcut(shortcut, null)
            }
        }

        bottomSheetView.findViewById<MaterialButton>(R.id.cheats).setOnClickListener {
            val action = CheatsFragmentDirections.actionGlobalCheatsFragment(holder.game.titleId)
            view.findNavController().navigate(action)
            bottomSheetDialog.dismiss()
        }

        fun getSaveDir(): DocumentFile? {
            val root = DocumentFile.fromTreeUri(LimeApplication.appContext, Uri.parse(userDirectory)) ?: return null

            return root.findFile("sdmc")
                ?.findFile("Nintendo 3DS")
                ?.findFile("00000000000000000000000000000000")
                ?.findFile("00000000000000000000000000000000")
                ?.findFile("title")
                ?.findFile(String.format("%016x", game.titleId).lowercase().substring(0, 8))
                ?.findFile(String.format("%016x", game.titleId).lowercase().substring(8))
                ?.findFile("data")
                ?.findFile("00000001")
        }

        // Better keep these separate
        fun getModsDir(): DocumentFile? {
            val root = DocumentFile.fromTreeUri(LimeApplication.appContext, Uri.parse(userDirectory)) ?: return null
            val loadDir = root.findFile("load") ?: root.createDirectory("load")
            val modsDir = loadDir?.findFile("mods") ?: loadDir?.createDirectory("mods")
            val titleId = String.format("%016X", game.titleId)
            return modsDir?.findFile(titleId) ?: modsDir?.createDirectory(titleId)
        }
        fun getTexturesDir(): DocumentFile? {
            val root = DocumentFile.fromTreeUri(LimeApplication.appContext, Uri.parse(userDirectory)) ?: return null
            val loadDir = root.findFile("load") ?: root.createDirectory("load")
            val texturesDir = loadDir?.findFile("textures") ?: loadDir?.createDirectory("textures")
            val titleId = String.format("%016X", game.titleId)
            return texturesDir?.findFile(titleId) ?: texturesDir?.createDirectory(titleId)
        }

        fun getAppDir(): DocumentFile? {
            var root = DocumentFile.fromTreeUri(LimeApplication.appContext, Uri.parse(userDirectory)) ?: return null

            val formattedPath = game.path
                .substringBeforeLast("/")
                .split("/")
                .filter { it.isNotEmpty() }

            for (component in formattedPath) {
                root = root.findFile(component) ?: return null
            }
            return root
        }

        fun getDLCAndUpdatesDir(isDLC: Boolean = false): DocumentFile? {
            val root = DocumentFile.fromTreeUri(LimeApplication.appContext, Uri.parse(userDirectory)) ?: return null
            val sysDir = if (isDLC) "0004008c" else "0004000e"
            return root.findFile("sdmc")
            ?.findFile("Nintendo 3DS")
            ?.findFile("00000000000000000000000000000000")
            ?.findFile("00000000000000000000000000000000")
            ?.findFile("title")
            ?.findFile(sysDir)
            ?.findFile(String.format("%016x", game.titleId).lowercase().substring(8))
            ?.findFile("content")
        }

        fun getExtraDir(): DocumentFile? {
            val root = DocumentFile.fromTreeUri(LimeApplication.appContext, Uri.parse(userDirectory)) ?: return null
            val extDataDir = root.findFile("sdmc")
                ?.findFile("Nintendo 3DS")
                ?.findFile("00000000000000000000000000000000")
                ?.findFile("00000000000000000000000000000000")
                ?.findFile("extdata")
                ?.findFile("00000000")
            val titleId = String.format("%016X", game.titleId).substring(8, 14).padStart(8, '0')
            return extDataDir?.findFile(titleId.uppercase()) ?: extDataDir?.findFile(titleId.lowercase())
        }

        val gameDir = game.path.substringBeforeLast("/")
        fun showOpenContextMenu(view: View, game: Game) {
            val popup = PopupMenu(view.context, view).apply {
            menuInflater.inflate(R.menu.game_context_menu_open, menu)
            menu.findItem(R.id.game_context_open_app).isEnabled = game.isInstalled
            menu.findItem(R.id.game_context_open_save_dir).isEnabled = getSaveDir() != null
            menu.findItem(R.id.game_context_open_dlc).isEnabled = getDLCAndUpdatesDir(isDLC = true) != null
            menu.findItem(R.id.game_context_open_updates).isEnabled = getDLCAndUpdatesDir(isDLC = false) != null
            menu.findItem(R.id.game_context_open_textures).isEnabled = getTexturesDir() != null
            if (getExtraDir() == null) {
                menu.removeItem(R.id.game_context_open_extra)
            }
            }

            popup.setOnMenuItemClickListener { menuItem ->
            val intent = Intent(Intent.ACTION_VIEW)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                .setType("*/*")

            when (menuItem.itemId) {
                R.id.game_context_open_app -> getAppDir()?.let { intent.data = it.uri }
                R.id.game_context_open_save_dir -> getSaveDir()?.let { intent.data = it.uri }
                R.id.game_context_open_dlc -> getDLCAndUpdatesDir(isDLC = true)?.let { intent.data = it.uri }
                R.id.game_context_open_updates -> getDLCAndUpdatesDir(isDLC = false)?.let { intent.data = it.uri }
                R.id.game_context_open_textures -> getTexturesDir()?.let { intent.data = it.uri }
                R.id.game_context_open_mods -> getModsDir()?.let { intent.data = it.uri }
                R.id.game_context_open_extra -> getExtraDir()?.let { intent.data = it.uri }
                else -> return@setOnMenuItemClickListener false
            }
            view.context.startActivity(intent)
            true
            }

            popup.show()
        }

        fun showUninstallContextMenu(view: View, game: Game) {
            val popup = PopupMenu(view.context, view).apply {
                menuInflater.inflate(R.menu.game_context_menu_uninstall, menu)
                menu.findItem(R.id.game_context_uninstall).isEnabled = game.isInstalled
                menu.findItem(R.id.game_context_uninstall_dlc).isEnabled = getDLCAndUpdatesDir(isDLC = true) != null
                menu.findItem(R.id.game_context_uninstall_updates).isEnabled = getDLCAndUpdatesDir(isDLC = false) != null
            }

            popup.setOnMenuItemClickListener { menuItem ->
                val uninstallAction: () -> Unit = {
                    when (menuItem.itemId) {
                        R.id.game_context_uninstall -> LimeApplication.documentsTree.deleteDocument(gameDir)
                        R.id.game_context_uninstall_dlc -> FileUtil.deleteDocument(getDLCAndUpdatesDir(isDLC = true)?.uri.toString())
                        R.id.game_context_uninstall_updates -> FileUtil.deleteDocument(getDLCAndUpdatesDir(isDLC = false)?.uri.toString())
                    }
                    ViewModelProvider(activity)[GamesViewModel::class.java].reloadGames(true)
                    bottomSheetDialog.dismiss()
                }

                if (menuItem.itemId in listOf(R.id.game_context_uninstall, R.id.game_context_uninstall_dlc, R.id.game_context_uninstall_updates)) {
                    IndeterminateProgressDialogFragment.newInstance(activity, R.string.uninstalling, false, uninstallAction)
                        .show(activity.supportFragmentManager, IndeterminateProgressDialogFragment.TAG)
                    true
                } else {
                    false
                }
            }

            popup.show()
        }

        bottomSheetView.findViewById<MaterialButton>(R.id.menu_button_open).setOnClickListener {
            showOpenContextMenu(it, game)
        }

        bottomSheetView.findViewById<MaterialButton>(R.id.menu_button_uninstall).setOnClickListener {
            showUninstallContextMenu(it, game)
        }

        val bottomSheetBehavior = bottomSheetDialog.getBehavior()
        bottomSheetBehavior.skipCollapsed = true
        bottomSheetBehavior.state = BottomSheetBehavior.STATE_EXPANDED

        bottomSheetDialog.show()
    }

    private fun isValidGame(extension: String): Boolean {
        return Game.badExtensions.stream()
            .noneMatch { extension == it.lowercase() }
    }

    private class DiffCallback : DiffUtil.ItemCallback<Game>() {
        override fun areItemsTheSame(oldItem: Game, newItem: Game): Boolean {
            return oldItem.titleId == newItem.titleId
        }

        override fun areContentsTheSame(oldItem: Game, newItem: Game): Boolean {
            return oldItem == newItem
        }
    }
}
