package dev.sensorium.app

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * The boundary between the device and the reasoning service.
 *
 * The workflow does not run on the phone, and this file is where that fact is made
 * explicit rather than buried. The phone records signals and asks questions; the nodes that
 * call a language model run off-device. Two consequences follow and both are deliberate:
 * the phone never holds an API key, and the exact prompts, model routing and validators
 * being demonstrated are the ones in this repository rather than a reimplementation that
 * could drift from them.
 *
 * The default address is loopback, reached over `adb reverse`, so a demo needs no network,
 * no tunnel and no server on the public internet. The journal text leaves the device only
 * when the person presses the button that says it will.
 */
class ApiClient(private val baseUrl: String) {

    data class Report(
        val markdown: String,
        val headline: String,
        val disagreement: String?,
        val insufficientData: Boolean,
        val suggestions: List<Suggestion>,
        val figures: List<Figure>,
        val runId: String,
    )

    data class Suggestion(val text: String, val sourceUrl: String?)

    /**
     * A figure with its significance verdict, which is three-valued on purpose.
     *
     * `null` means the window could not answer whether the slope differs from zero, and it
     * is kept distinct from `false`, which means it was tested and is indistinguishable
     * from noise. Folding the two together would turn "we could not test this" into "we
     * tested this and there is nothing there" — a claim the data does not support.
     */
    data class Figure(val name: String, val value: String, val significant: Boolean?)

    class ServiceError(message: String) : Exception(message)

    suspend fun health(): String = withContext(Dispatchers.IO) {
        request("GET", "/health", null).optString("status", "unknown")
    }

    /**
     * Run the workflow over this device's history and this week's answers.
     *
     * Failures are surfaced, never smoothed. If the service refuses to produce a report -
     * which it does when a node cannot satisfy its contract - the app says so and shows
     * why. An app that silently degrades to a plausible-looking screen would undo the one
     * property the whole system is built around.
     */
    suspend fun analyse(deviceSlice: JSONObject, conversation: JSONArray): Report =
        withContext(Dispatchers.IO) {
            val body = JSONObject().apply {
                put("device_slice", deviceSlice)
                put("conversation", conversation)
            }
            parse(request("POST", "/analyse", body))
        }

    private fun parse(json: JSONObject): Report {
        val suggestions = json.optJSONArray("suggestions") ?: JSONArray()
        val figures = json.optJSONArray("figures") ?: JSONArray()
        return Report(
            markdown = json.optString("report_markdown", ""),
            headline = json.optString("headline", ""),
            disagreement = json.stringOrNull("disagreement"),
            insufficientData = json.optBoolean("insufficient_data", false),
            suggestions = (0 until suggestions.length()).map { index ->
                val item = suggestions.getJSONObject(index)
                Suggestion(
                    text = item.optString("text"),
                    sourceUrl = item.stringOrNull("source_url"),
                )
            },
            figures = (0 until figures.length()).map { index ->
                val item = figures.getJSONObject(index)
                Figure(
                    name = item.optString("name"),
                    value = item.optString("value"),
                    significant = if (item.isNull("significant")) {
                        null
                    } else {
                        item.optBoolean("significant")
                    },
                )
            },
            runId = json.optString("run_id", ""),
        )
    }

    /**
     * `optString` renders a JSON null as the four-character string "null", which would put
     * the word null on screen where a citation URL belongs. An explicit null is a real
     * answer here — Node 6 leaves `source_url` null when no retrieved source covers a
     * suggestion, and that absence is deliberate — so it has to survive parsing as one.
     */
    private fun JSONObject.stringOrNull(key: String): String? =
        if (isNull(key)) null else optString(key).ifBlank { null }

    private fun request(method: String, path: String, body: JSONObject?): JSONObject {
        val connection = (URL(baseUrl + path).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 10_000
            readTimeout = 600_000
            setRequestProperty("Content-Type", "application/json")
            doInput = true
        }

        try {
            if (body != null) {
                connection.doOutput = true
                connection.outputStream.use { it.write(body.toString().toByteArray()) }
            }

            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val text = stream?.bufferedReader()?.use { it.readText() }.orEmpty()

            if (code !in 200..299) {
                val detail = runCatching { JSONObject(text).optString("detail") }.getOrNull()
                throw ServiceError(detail?.ifBlank { null } ?: "HTTP $code")
            }
            return JSONObject(text)
        } finally {
            connection.disconnect()
        }
    }

    companion object {
        /** Loopback, reached with `adb reverse tcp:8765 tcp:8765`. */
        const val DEFAULT_BASE_URL = "http://127.0.0.1:8765"
    }
}
