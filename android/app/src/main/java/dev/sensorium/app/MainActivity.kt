package dev.sensorium.app

import android.os.Bundle
import android.text.method.LinkMovementMethod
import android.view.View
import android.widget.ArrayAdapter
import androidx.appcompat.app.AppCompatActivity
import androidx.core.text.HtmlCompat
import androidx.lifecycle.lifecycleScope
import dev.sensorium.app.databinding.ActivityMainBinding
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject

/**
 * One screen, in the order the workflow runs: what the device reports, what the person
 * says, and only then what the system concluded.
 *
 * The ordering is the argument. A person can see every reading before any of it is
 * recorded, see the questions before answering them, and see the refusal boundary at the
 * top of the result rather than in a footer. Nothing on this screen is computed here:
 * every number displayed arrives from the statistics engine with its significance flag
 * still attached, and every sentence arrives from a node whose output was validated
 * before it was returned.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var store: SignalStore
    private val api = ApiClient(ApiClient.DEFAULT_BASE_URL)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        store = SignalStore(this)

        binding.reportMarkdown.movementMethod = LinkMovementMethod.getInstance()

        binding.driftSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            listOf("rising trend", "no trend"),
        )

        binding.refreshButton.setOnClickListener { showLiveSignals() }
        binding.recordButton.setOnClickListener {
            store.recordNow(this)
            showHistory()
            toast("Recorded this device's current readings")
        }
        binding.seedButton.setOnClickListener {
            val drift = if (binding.driftSpinner.selectedItemPosition == 0) {
                SignalStore.Drift.RISING
            } else {
                SignalStore.Drift.FLAT
            }
            store.seedHistory(this, drift = drift)
            showHistory()
            toast("Seeded 4 weeks, anchored on this device's real readings")
        }
        binding.clearButton.setOnClickListener {
            store.clear()
            showHistory()
        }
        binding.analyseButton.setOnClickListener { analyse() }

        showLiveSignals()
        showHistory()
        checkService()
    }

    private fun showLiveSignals() {
        binding.signalsTable.removeAllViews()
        SignalReader.describe(this).forEach { (label, value) ->
            binding.signalsTable.addView(Rows.build(this, label, value))
        }
    }

    private fun showHistory() {
        val recorded = store.recordedCount()
        val seeded = store.seededCount()
        binding.historySummary.text = getString(R.string.history_summary, recorded, seeded)
        binding.analyseButton.isEnabled = recorded + seeded > 0
    }

    private fun checkService() {
        lifecycleScope.launch {
            val text = runCatching { api.health() }
                .fold(
                    onSuccess = { getString(R.string.service_ready) },
                    onFailure = { getString(R.string.service_missing) },
                )
            binding.serviceStatus.text = text
        }
    }

    /**
     * The one place journal text leaves the device, behind the button that says so.
     *
     * Errors are shown as errors. When a node cannot satisfy its contract the service
     * refuses to produce a report, and this screen prints that refusal instead of falling
     * back to something presentable. A demo that hides its failures is demonstrating the
     * failure mode rather than the system.
     */
    private fun analyse() {
        val conversation = collectConversation()
        if (conversation.length() == 0) {
            toast("Answer at least one question first")
            return
        }

        setBusy(true)
        binding.resultCard.visibility = View.GONE
        binding.errorText.visibility = View.GONE

        lifecycleScope.launch {
            runCatching { api.analyse(store.deviceSlice(PROFILE_ID), conversation) }
                .onSuccess { render(it) }
                .onFailure { failure ->
                    binding.errorText.visibility = View.VISIBLE
                    binding.errorText.text =
                        getString(R.string.analysis_refused, failure.message ?: "unknown")
                }
            setBusy(false)
        }
    }

    private fun collectConversation(): JSONArray {
        val questions = resources.getStringArray(R.array.checkin_questions)
        val answers = listOf(binding.answerOne, binding.answerTwo, binding.answerThree)
        val conversation = JSONArray()

        answers.forEachIndexed { index, field ->
            val text = field.text.toString().trim()
            if (text.isNotEmpty()) {
                conversation.put(turn("agent", questions[index]))
                conversation.put(turn("user", text))
            }
        }
        return conversation
    }

    private fun turn(role: String, text: String) = JSONObject().apply {
        put("role", role)
        put("text", text)
    }

    private fun render(report: ApiClient.Report) {
        binding.resultCard.visibility = View.VISIBLE

        binding.verdictText.text = when {
            report.insufficientData -> getString(R.string.verdict_no_trend)
            else -> getString(R.string.verdict_trend)
        }
        binding.headlineText.text = report.headline

        binding.disagreementCard.visibility =
            if (report.disagreement.isNullOrBlank()) View.GONE else View.VISIBLE
        binding.disagreementText.text = report.disagreement.orEmpty()

        binding.figuresTable.removeAllViews()
        report.figures.forEach { figure ->
            val marker = when (figure.significant) {
                true -> getString(R.string.significant)
                false -> getString(R.string.not_significant)
                null -> getString(R.string.untested)
            }
            binding.figuresTable.addView(
                Rows.build(this, figure.name, "${figure.value}  $marker")
            )
        }

        binding.suggestionsText.text = report.suggestions.joinToString("\n\n") { suggestion ->
            val source = suggestion.sourceUrl?.let { "\n$it" }
                ?: "\n${getString(R.string.no_source)}"
            "• ${suggestion.text}$source"
        }

        binding.reportMarkdown.text = HtmlCompat.fromHtml(
            Markdown.toHtml(report.markdown), HtmlCompat.FROM_HTML_MODE_COMPACT
        )
        binding.runIdText.text = getString(R.string.run_id, report.runId)
    }

    private fun setBusy(busy: Boolean) {
        binding.progress.visibility = if (busy) View.VISIBLE else View.GONE
        binding.analyseButton.isEnabled = !busy
        binding.analyseButton.text = getString(
            if (busy) R.string.analysing else R.string.analyse
        )
    }

    private fun toast(message: String) {
        android.widget.Toast.makeText(this, message, android.widget.Toast.LENGTH_SHORT).show()
    }

    private companion object {
        const val PROFILE_ID = "device-local"
    }
}
