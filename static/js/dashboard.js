// WattWise dashboard chart: daily usage line, anomalous days in red.
(function () {
  var el = document.getElementById("usageChart");
  if (!el || !window.WATTWISE_CHART || typeof Chart === "undefined") return;
  var d = window.WATTWISE_CHART;
  var pointColors = d.anomaly.map(function (a) { return a ? "#E5484D" : "#FFB020"; });
  var pointRadii = d.anomaly.map(function (a) { return a ? 6 : 3; });
  new Chart(el, {
    type: "line",
    data: {
      labels: d.labels,
      datasets: [{
        label: "Units (kWh) per day",
        data: d.units,
        borderColor: "#F5A623",
        backgroundColor: "rgba(245, 166, 35, 0.15)",
        fill: true,
        tension: 0.25,
        pointBackgroundColor: pointColors,
        pointBorderColor: pointColors,
        pointRadius: pointRadii
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#D8D2C4" } },
        tooltip: {
          callbacks: {
            afterLabel: function (ctx) {
              return d.anomaly[ctx.dataIndex] ? "⚠ Abnormal day" : "Normal day";
            }
          }
        }
      },
      scales: {
        x: { ticks: { color: "#9A937F", maxTicksLimit: 12 }, grid: { color: "#2A251C" } },
        y: { ticks: { color: "#9A937F" }, grid: { color: "#2A251C" },
             title: { display: true, text: "Units (kWh)", color: "#9A937F" } }
      }
    }
  });
})();
