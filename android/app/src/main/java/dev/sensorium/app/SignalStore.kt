package dev.sensorium.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import kotlin.random.Random

/**
 * The recorded history, held in the app's own storage and nowhere else.
 *
 * A trend needs weeks. This class is what turns a single reading into a series, and it is
 * a plain JSON file in `filesDir` rather than a database because the whole store is a few
 * hundred numbers and because a file can be read, exported and deleted by the person it
 * describes without any tooling.
 *
 * Seeded history is kept in a separate list from recorded history and is labelled as such
 * everywhere it surfaces. A demo needs four weeks of readings and a demo cannot wait four
 * weeks, but a seeded reading that could be mistaken for a measured one would make every
 * number downstream a claim about nothing. The two are never merged silently: the report
 * screen says how many of each went in.
 */
class SignalStore(context: Context) {

    private val file = File(context.filesDir, "signals.json")

    data class Sample(val events: List<SignalEvent>, val seeded: Boolean)

    fun load(): List<Sample> {
        if (!file.exists()) return emptyList()
        return runCatching {
            val root = JSONArray(file.readText())
            (0 until root.length()).map { index ->
                val entry = root.getJSONObject(index)
                val events = entry.getJSONArray("events")
                Sample(
                    events = (0 until events.length()).map { position ->
                        val event = events.getJSONObject(position)
                        SignalEvent(
                            signal = event.getString("signal"),
                            timestamp = event.getString("ts"),
                            value = event.getDouble("value"),
                        )
                    },
                    seeded = entry.optBoolean("seeded", false),
                )
            }
        }.getOrDefault(emptyList())
    }

    private fun save(samples: List<Sample>) {
        val root = JSONArray()
        samples.forEach { sample ->
            root.put(JSONObject().apply {
                put("events", sample.events.toJsonArray())
                put("seeded", sample.seeded)
            })
        }
        file.writeText(root.toString())
    }

    /** Record what the device reports right now, stamped with the current time. */
    fun recordNow(context: Context) {
        val stamp = LocalDateTime.now(ZoneOffset.UTC).format(TIMESTAMP)
        val events = SignalReader.read(context).map { (signal, value) ->
            SignalEvent(signal, stamp, value)
        }
        save(load() + Sample(events, seeded = false))
    }

    fun clear() = save(emptyList())

    fun recordedCount() = load().count { !it.seeded }

    fun seededCount() = load().count { it.seeded }

    /**
     * Four weeks of plausible history, anchored on this device's actual current readings.
     *
     * The drift is applied to the real values this phone reports, so the series ends where
     * the device genuinely is today rather than at an invented number. Whether a trend is
     * statistically significant is still decided by the engine on the resulting series, not
     * chosen here - this method has no idea what verdict it is producing, which is the only
     * way a seeded demo can still be an honest one.
     */
    fun seedHistory(context: Context, weeks: Int = 4, drift: Drift) {
        val today = SignalReader.read(context)
        val random = Random(SEED)
        val samples = mutableListOf<Sample>()

        val steps = weeks * SAMPLES_PER_WEEK
        for (index in 0 until steps) {
            val daysAgo = ((steps - 1 - index) * 7.0 / SAMPLES_PER_WEEK).toInt()
            val stamp = LocalDate.now()
                .minusDays(daysAgo.toLong())
                .atTime(19, 15)
                .format(TIMESTAMP)

            val progress = index.toDouble() / (steps - 1).coerceAtLeast(1)
            val events = today.mapNotNull { (signal, current) ->
                val value = when (signal) {
                    "volume", "brightness", "font_scale" ->
                        drifted(current, progress, drift, signal, random)
                    else -> current
                }
                SignalEvent(signal, stamp, value)
            }
            samples += Sample(events, seeded = true)
        }
        save(load().filter { !it.seeded } + samples)
    }

    /**
     * Walk a signal backwards from today's real value by the requested amount, with noise.
     *
     * The noise is not decoration. Without it every point sits exactly on the line, the
     * regression fits perfectly, and a significance test on a perfect fit tells you only
     * that you generated a perfect fit.
     */
    private fun drifted(
        current: Double, progress: Double, drift: Drift, signal: String, random: Random,
    ): Double {
        val proportion = when (drift) {
            Drift.RISING -> 0.25
            Drift.FLAT -> 0.0
        }
        val start = current / (1 + proportion)
        val noiseScale = if (signal == "font_scale") 0.01 else 0.04
        val noise = (random.nextDouble() - 0.5) * 2 * noiseScale * current
        return ((start + (current - start) * progress) + noise).coerceAtLeast(0.0)
    }

    /**
     * Which shape of history to generate.
     *
     * FLAT exists so the demo can show the system declining to report a trend. A workflow
     * that only ever has a finding to announce has not been shown to be capable of saying
     * there isn't one.
     */
    enum class Drift { RISING, FLAT }

    /** The device slice, in exactly the shape `stats.engine.compute` expects. */
    fun deviceSlice(profileId: String): JSONObject {
        val samples = load()
        val events = samples.flatMap { it.events }
        val dates = events.map { it.timestamp.substring(0, 10) }.sorted()
        return JSONObject().apply {
            put("events", events.toJsonArray())
            put("profile_id", profileId)
            put("window", JSONObject().apply {
                put("start", dates.firstOrNull() ?: LocalDate.now().toString())
                put("end", dates.lastOrNull() ?: LocalDate.now().toString())
            })
        }
    }

    private companion object {
        val TIMESTAMP: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss'+00:00'")
        const val SAMPLES_PER_WEEK = 4
        const val SEED = 20260824L
    }
}
