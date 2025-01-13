// Copyright 2024 Mandarine Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.dialogs

import android.content.Context
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.PopupMenu
import android.widget.Toast
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import io.github.borked3ds.android.Borked3DSApplication
import io.github.borked3ds.android.R
import io.github.borked3ds.android.databinding.*
import io.github.borked3ds.android.utils.CompatUtils
import io.github.borked3ds.android.utils.NetPlayManager

class NetPlayDialog(context: Context) : BaseSheetDialog(context) {
    private lateinit var adapter: NetPlayAdapter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (NetPlayManager.netPlayIsJoined()) {
            DialogMultiplayerBinding.inflate(layoutInflater).apply {
                setContentView(root)
                adapter = NetPlayAdapter()
                listMultiplayer.layoutManager = LinearLayoutManager(context)
                listMultiplayer.adapter = adapter
                adapter.loadMultiplayerMenu()
                btnLeave.setOnClickListener {
                    NetPlayManager.clearChat()
                    NetPlayManager.netPlayLeaveRoom()
                    dismiss()
                }
                btnChat.setOnClickListener {
                    ChatDialog(context).show()
                }
            }
        } else {
            DialogMultiplayerInitialBinding.inflate(layoutInflater).apply {
                setContentView(root)
                btnCreate.setOnClickListener {
                    showNetPlayInputDialog(true)
                    dismiss()
                }
                btnJoin.setOnClickListener {
                    showNetPlayInputDialog(false)
                    dismiss()
                }
            }
        }
    }

    data class NetPlayItems(
        val option: Int,
        val name: String,
        val type: Int,
        val id: Int = 0
    ) {
        companion object {
            const val MULTIPLAYER_ROOM_TEXT = 1
            const val MULTIPLAYER_ROOM_MEMBER = 2
            const val MULTIPLAYER_SEPARATOR = 3
            const val MULTIPLAYER_ROOM_COUNT = 4
            const val TYPE_BUTTON = 0
            const val TYPE_TEXT = 1
            const val TYPE_SEPARATOR = 2
        }
    }

    inner class NetPlayAdapter : RecyclerView.Adapter<NetPlayAdapter.NetPlayViewHolder>() {
        private val netPlayItems = mutableListOf<NetPlayItems>()

        abstract inner class NetPlayViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView),
            View.OnClickListener {
            init {
                itemView.setOnClickListener(this)
            }

            abstract fun bind(item: NetPlayItems)
        }

        inner class TextViewHolder(private val binding: ItemTextNetplayBinding) :
            NetPlayViewHolder(binding.root) {
            private lateinit var netPlayItem: NetPlayItems

            override fun onClick(clicked: View) {}

            override fun bind(item: NetPlayItems) {
                netPlayItem = item
                binding.itemTextNetplayName.text = item.name
                binding.itemIcon.apply {
                    val iconRes = when (item.option) {
                        NetPlayItems.MULTIPLAYER_ROOM_TEXT -> R.drawable.ic_system
                        NetPlayItems.MULTIPLAYER_ROOM_COUNT -> R.drawable.ic_joined
                        else -> 0
                    }
                    visibility = if (iconRes != 0) {
                        setImageResource(iconRes)
                        View.VISIBLE
                    } else View.GONE
                }
            }
        }

        inner class ButtonViewHolder(private val binding: ItemButtonNetplayBinding) :
            NetPlayViewHolder(binding.root) {
            private lateinit var netPlayItems: NetPlayItems
            private val isModerator = NetPlayManager.netPlayIsModerator()

            init {
                binding.itemButtonMore.apply {
                    visibility = View.VISIBLE
                    setOnClickListener { showPopupMenu(it) }
                }
            }

            override fun onClick(clicked: View) {}

            private fun showPopupMenu(view: View) {
                PopupMenu(view.context, view).apply {
                    inflate(R.menu.menu_netplay_member)
                    menu.findItem(R.id.action_kick).isEnabled = isModerator &&
                            netPlayItems.name != NetPlayManager.getUsername(context)
                    setOnMenuItemClickListener { item ->
                        if (item.itemId == R.id.action_kick) {
                            NetPlayManager.netPlayKickUser(netPlayItems.name)
                            true
                        } else false
                    }
                    show()
                }
            }

            override fun bind(item: NetPlayItems) {
                netPlayItems = item
                binding.itemButtonNetplayName.text = netPlayItems.name
            }
        }

        fun loadMultiplayerMenu() {
            val infos = NetPlayManager.netPlayRoomInfo()
            if (infos.isNotEmpty()) {
                val roomInfo = infos[0].split("|")
                netPlayItems.add(
                    NetPlayItems(
                        NetPlayItems.MULTIPLAYER_ROOM_TEXT,
                        roomInfo[0],
                        NetPlayItems.TYPE_TEXT
                    )
                )
                netPlayItems.add(
                    NetPlayItems(
                        NetPlayItems.MULTIPLAYER_ROOM_COUNT,
                        "${infos.size - 1}/${roomInfo[1]}",
                        NetPlayItems.TYPE_TEXT
                    )
                )
                netPlayItems.add(
                    NetPlayItems(
                        NetPlayItems.MULTIPLAYER_SEPARATOR,
                        "",
                        NetPlayItems.TYPE_SEPARATOR
                    )
                )
                for (i in 1 until infos.size) {
                    netPlayItems.add(
                        NetPlayItems(
                            NetPlayItems.MULTIPLAYER_ROOM_MEMBER,
                            infos[i],
                            NetPlayItems.TYPE_BUTTON
                        )
                    )
                }
            }
        }

        override fun getItemViewType(position: Int) = netPlayItems[position].type

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): NetPlayViewHolder {
            val inflater = LayoutInflater.from(parent.context)
            return when (viewType) {
                NetPlayItems.TYPE_TEXT -> TextViewHolder(
                    ItemTextNetplayBinding.inflate(
                        inflater,
                        parent,
                        false
                    )
                )

                NetPlayItems.TYPE_BUTTON -> ButtonViewHolder(
                    ItemButtonNetplayBinding.inflate(
                        inflater,
                        parent,
                        false
                    )
                )

                NetPlayItems.TYPE_SEPARATOR -> object : NetPlayViewHolder(
                    inflater.inflate(
                        R.layout.item_separator_netplay,
                        parent,
                        false
                    )
                ) {
                    override fun bind(item: NetPlayItems) {}
                    override fun onClick(clicked: View) {}
                }

                else -> throw IllegalStateException("Unsupported view type")
            }
        }

        override fun onBindViewHolder(holder: NetPlayViewHolder, position: Int) {
            holder.bind(netPlayItems[position])
        }

        override fun getItemCount() = netPlayItems.size
    }

    private fun showNetPlayInputDialog(isCreateRoom: Boolean) {
        val activity = CompatUtils.findActivity(context)
        val dialog = BaseSheetDialog(activity)
        val binding = DialogMultiplayerRoomBinding.inflate(LayoutInflater.from(activity))
        dialog.setContentView(binding.root)

        binding.textTitle.text = activity.getString(
            if (isCreateRoom) R.string.multiplayer_create_room
            else R.string.multiplayer_join_room
        )

        binding.ipAddress.setText(
            if (isCreateRoom) NetPlayManager.getIpAddressByWifi(activity)
            else NetPlayManager.getRoomAddress(activity)
        )
        binding.ipPort.setText(NetPlayManager.getRoomPort(activity))
        binding.username.setText(NetPlayManager.getUsername(activity))

        binding.roomName.visibility = if (isCreateRoom) View.VISIBLE else View.GONE
        binding.maxPlayersContainer.visibility = if (isCreateRoom) View.VISIBLE else View.GONE
        binding.maxPlayersLabel.text = context.getString(
            R.string.multiplayer_max_players_value,
            binding.maxPlayers.value.toInt()
        )

        binding.maxPlayers.addOnChangeListener { _, value, _ ->
            binding.maxPlayersLabel.text =
                context.getString(R.string.multiplayer_max_players_value, value.toInt())
        }

        binding.btnConfirm.setOnClickListener {
            binding.btnConfirm.isEnabled = false
            binding.btnConfirm.text = activity.getString(R.string.disabled_button_text)

            val ipAddress = binding.ipAddress.text.toString()
            val username = binding.username.text.toString()
            val portStr = binding.ipPort.text.toString()
            val password = binding.password.text.toString()
            val port = portStr.toIntOrNull() ?: run {
                Toast.makeText(activity, R.string.multiplayer_port_invalid, Toast.LENGTH_LONG)
                    .show()
                binding.btnConfirm.isEnabled = true
                binding.btnConfirm.text = activity.getString(R.string.original_button_text)
                return@setOnClickListener
            }
            val roomName = binding.roomName.text.toString()
            val maxPlayers = binding.maxPlayers.value.toInt()

            if (isCreateRoom && (roomName.length !in 3..20)) {
                Toast.makeText(activity, R.string.multiplayer_room_name_invalid, Toast.LENGTH_LONG)
                    .show()
                binding.btnConfirm.isEnabled = true
                binding.btnConfirm.text = activity.getString(R.string.original_button_text)
                return@setOnClickListener
            }

            if (ipAddress.length < 7 || username.length < 5) {
                Toast.makeText(activity, R.string.multiplayer_input_invalid, Toast.LENGTH_LONG)
                    .show()
                binding.btnConfirm.isEnabled = true
                binding.btnConfirm.text = activity.getString(R.string.original_button_text)
            } else {
                Handler(Looper.getMainLooper()).post {
                    val result = if (isCreateRoom) {
                        NetPlayManager.netPlayCreateRoom(
                            ipAddress,
                            port,
                            username,
                            password,
                            roomName,
                            maxPlayers
                        )
                    } else {
                        NetPlayManager.netPlayJoinRoom(ipAddress, port, username, password)
                    }

                    if (result == 0) {
                        NetPlayManager.setUsername(activity, username)
                        NetPlayManager.setRoomPort(activity, portStr)
                        if (!isCreateRoom) NetPlayManager.setRoomAddress(activity, ipAddress)
                        Toast.makeText(
                            Borked3DSApplication.appContext,
                            if (isCreateRoom) R.string.multiplayer_create_room_success
                            else R.string.multiplayer_join_room_success,
                            Toast.LENGTH_LONG
                        ).show()
                        dialog.dismiss()
                    } else {
                        Toast.makeText(
                            activity,
                            R.string.multiplayer_could_not_connect,
                            Toast.LENGTH_LONG
                        ).show()
                        binding.btnConfirm.isEnabled = true
                        binding.btnConfirm.text = activity.getString(R.string.original_button_text)
                    }
                }
            }
        }

        dialog.show()
    }
}
