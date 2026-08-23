package dev.sensorium.app

import android.content.Context
import android.media.AudioManager
import android.provider.Settings
import org.json.JSONArray
import org.json.JSONObject

/**
 * Node 0 - passive signal capture.
 *
 * Every reading here is available without a runtime permission prompt and without any
 * accessibility or usage-stats grant. That is a design constraint, not a convenience: a
 * hearing- and vision-trend app that asks for privileged access to work is one most people
 * will decline, and the signals that survive the constraint turned out to be enough.
 *
 * What is deliberately absent is as important as what is here. Foreground time appears in
 * the workflow's data model but not in this reader, because on Android it requires the
 * PACKAGE_USAGE_STATS special access. The engine treats a missing signal as missing rather
 * than as zero, so omitting it costs a figure and corrupts nothing.
 *
 * Nothing in this file interprets a reading. It records what the device reports and hands
 * it to a statistics engine that is not a language model; the numbers a person eventually
 * sees are computed there, which is why no model is ever in a position to invent one.
 */
object SignalReader {

    /** One reading of every signal this device can report without a permission prompt. */
    fun read(context: Context): Map<String, Double> {
        val out = LinkedHashMap<String, Double>()

        volumeSteps(context)?.let { out["volume"] = it }
        brightnessLevel(context)?.let { out["brightness"] = it }
        out["font_scale"] = context.resources.configuration.fontScale.toDouble()
        captionsOn(context)?.let { out["caption"] = it }
        autoBrightnessOn(context)?.let { out["brightness_mode"] = it }

        return out
    }

    /**
     * Media stream volume, in steps.
     *
     * Reported as raw steps rather than as a percentage because the engine's unit for this
     * signal is a step, and because the number of steps differs between devices. Converting
     * to a percentage here would silently make two phones' readings look comparable when
     * they are not.
     */
    private fun volumeSteps(context: Context): Double? {
        val audio = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager ?: return null
        return audio.getStreamVolume(AudioManager.STREAM_MUSIC).toDouble()
    }

    /** Screen brightness on this device's own 0-255 scale. */
    private fun brightnessLevel(context: Context): Double? = try {
        Settings.System.getInt(context.contentResolver, Settings.System.SCREEN_BRIGHTNESS).toDouble()
    } catch (_: Settings.SettingNotFoundException) {
        null
    }

    /** Whether the person has turned closed captions on. A toggle, so the engine treats it as one. */
    private fun captionsOn(context: Context): Double? = try {
        Settings.Secure.getInt(context.contentResolver, CAPTIONING_ENABLED).toDouble()
    } catch (_: Settings.SettingNotFoundException) {
        null
    }

    private fun autoBrightnessOn(context: Context): Double? = try {
        val mode = Settings.System.getInt(
            context.contentResolver, Settings.System.SCREEN_BRIGHTNESS_MODE
        )
        if (mode == Settings.System.SCREEN_BRIGHTNESS_MODE_AUTOMATIC) 1.0 else 0.0
    } catch (_: Settings.SettingNotFoundException) {
        null
    }

    /** Maximum media volume steps on this device, for display only - never sent as a figure. */
    fun maxVolumeSteps(context: Context): Int {
        val audio = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager ?: return 0
        return audio.getStreamMaxVolume(AudioManager.STREAM_MUSIC)
    }

    /**
     * Human-readable rows for the "what this device reports right now" panel.
     *
     * The panel exists so a person can see exactly what is being recorded about them before
     * anything is recorded. A privacy claim that cannot be inspected is a promise; this one
     * is a screen.
     */
    fun describe(context: Context): List<Pair<String, String>> {
        val readings = read(context)
        val rows = mutableListOf<Pair<String, String>>()

        readings["volume"]?.let {
            rows += "Media volume" to "${it.toInt()} of ${maxVolumeSteps(context)} steps"
        }
        readings["brightness"]?.let {
            rows += "Screen brightness" to "${it.toInt()} of 255"
        }
        readings["font_scale"]?.let {
            rows += "Font scale" to "${"%.2f".format(it)}x"
        }
        readings["caption"]?.let {
            rows += "Closed captions" to if (it > 0.5) "on" else "off"
        }
        readings["brightness_mode"]?.let {
            rows += "Adaptive brightness" to if (it > 0.5) "on" else "off"
        }
        return rows
    }

    /** Not exposed as a constant on Settings.Secure until later API levels. */
    private const val CAPTIONING_ENABLED = "accessibility_captioning_enabled"
}

/** One timestamped reading of one signal, in the shape the statistics engine consumes. */
data class SignalEvent(val signal: String, val timestamp: String, val value: Double) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("signal", signal)
        put("ts", timestamp)
        put("value", value)
    }
}

fun List<SignalEvent>.toJsonArray(): JSONArray =
    JSONArray().also { array -> forEach { array.put(it.toJson()) } }
