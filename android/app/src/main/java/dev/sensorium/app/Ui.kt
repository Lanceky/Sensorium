package dev.sensorium.app

import android.content.Context
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat

/** A label-and-value row, built in code so the signal panel can grow with the device. */
object Rows {
    fun build(context: Context, label: String, value: String): LinearLayout =
        LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            val pad = (8 * context.resources.displayMetrics.density).toInt()
            setPadding(0, pad / 2, 0, pad / 2)

            addView(TextView(context).apply {
                text = label
                setTextColor(ContextCompat.getColor(context, R.color.muted))
                layoutParams = LinearLayout.LayoutParams(0, -2, 1f)
            })
            addView(TextView(context).apply {
                text = value
                gravity = Gravity.END
                setTextColor(ContextCompat.getColor(context, R.color.ink))
                layoutParams = LinearLayout.LayoutParams(0, -2, 1f)
            })
        }
}

/**
 * Just enough Markdown to render Node 10's report.
 *
 * Node 10 emits Markdown because a doctor-shareable report should be plain text a person
 * can paste anywhere, not a proprietary blob. This converts the small subset that node
 * actually produces — headings, bold, bullets, links — and escapes everything else. It is
 * intentionally not a general Markdown engine: an incomplete renderer that drops an unknown
 * construct is safer here than a clever one that reinterprets the text of a health report.
 */
object Markdown {

    fun toHtml(markdown: String): String {
        val out = StringBuilder()
        markdown.lines().forEach { line ->
            val escaped = escape(line.trim())
            when {
                escaped.isEmpty() -> out.append("<br>")
                escaped.startsWith("### ") -> out.append("<b>${inline(escaped.drop(4))}</b><br>")
                escaped.startsWith("## ") -> out.append("<b>${inline(escaped.drop(3))}</b><br>")
                escaped.startsWith("# ") -> out.append("<b>${inline(escaped.drop(2))}</b><br>")
                escaped.startsWith("- ") -> out.append("• ${inline(escaped.drop(2))}<br>")
                escaped.startsWith("* ") -> out.append("• ${inline(escaped.drop(2))}<br>")
                else -> out.append("${inline(escaped)}<br>")
            }
        }
        return out.toString()
    }

    private fun escape(text: String) = text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")

    private fun inline(text: String): String {
        var result = BOLD.replace(text) { "<b>${it.groupValues[1]}</b>" }
        result = LINK.replace(result) { "<a href=\"${it.groupValues[2]}\">${it.groupValues[1]}</a>" }
        result = BARE_URL.replace(result) { "<a href=\"${it.value}\">${it.value}</a>" }
        return result
    }

    private val BOLD = Regex("""\*\*(.+?)\*\*""")
    private val LINK = Regex("""\[([^\]]+)]\(([^)]+)\)""")
    private val BARE_URL = Regex("""(?<!["=>])\bhttps?://[^\s<]+""")
}
