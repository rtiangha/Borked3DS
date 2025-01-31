// Copyright 2023 Citra Emulator Project
// Copyright 2024 Borked3DS Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

package io.github.borked3ds.android.viewmodel

import android.view.Surface
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import io.github.borked3ds.android.NativeLibrary
import io.github.borked3ds.android.viewmodel.EmulationState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class EmulationViewModel(private val savedStateHandle: SavedStateHandle) : ViewModel() {
    private var emulationState: EmulationState? = null
    var currentSurface: Surface? = null

    fun initializeEmulationState(gamePath: String) {
        if (emulationState == null) {
            emulationState = EmulationState(gamePath)
        }
    }

    fun getEmulationState(): EmulationState {
        return emulationState ?: throw IllegalStateException("EmulationState not initialized")
    }

    // Handle surface changes
    fun onSurfaceChanged(surface: Surface?) {
        currentSurface = surface
        emulationState?.newSurface(surface)
    }

    fun clearSurface() {
        currentSurface = null
        emulationState?.clearSurface()
    }

    val emulationStarted get() = _emulationStarted.asStateFlow()
    private val _emulationStarted = MutableStateFlow(false)

    val shaderProgress get() = _shaderProgress.asStateFlow()
    private val _shaderProgress = MutableStateFlow(0)

    val totalShaders get() = _totalShaders.asStateFlow()
    private val _totalShaders = MutableStateFlow(0)

    val shaderMessage get() = _shaderMessage.asStateFlow()
    private val _shaderMessage = MutableStateFlow("")

    fun setShaderProgress(progress: Int) {
        _shaderProgress.value = progress
    }

    fun setTotalShaders(max: Int) {
        _totalShaders.value = max
    }

    fun setShaderMessage(msg: String) {
        _shaderMessage.value = msg
    }

    fun updateProgress(msg: String, progress: Int, max: Int) {
        setShaderMessage(msg)
        setShaderProgress(progress)
        setTotalShaders(max)
    }

    fun setEmulationStarted(started: Boolean) {
        _emulationStarted.value = started
    }
}
