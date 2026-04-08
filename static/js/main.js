document.addEventListener("DOMContentLoaded", () => {
  const root = document.documentElement;
  const storedTheme = localStorage.getItem("sap-theme") || "light";
  root.setAttribute("data-theme", storedTheme);
  document.body.classList.toggle("dark-mode", storedTheme === "dark");

  const themeToggle = document.querySelectorAll("#themeToggle");
  themeToggle.forEach((button) => {
    button.addEventListener("click", () => {
      const nextTheme = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", nextTheme);
      document.body.classList.toggle("dark-mode", nextTheme === "dark");
      localStorage.setItem("sap-theme", nextTheme);
    });
  });

  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebarToggle");
  if (sidebar && sidebarToggle) {
    sidebarToggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }

  document.querySelectorAll(".toggle-password").forEach((button) => {
    button.addEventListener("click", () => {
      const targetId = button.getAttribute("data-target");
      const input = document.getElementById(targetId);
      if (!input) {
        return;
      }
      const isPassword = input.type === "password";
      input.type = isPassword ? "text" : "password";
      button.textContent = isPassword ? "Hide" : "Show";
    });
  });

  const predictionForm = document.getElementById("predictionForm");
  const spinner = document.getElementById("predictionSpinner");
  const resultDisplay = document.getElementById("resultDisplay");
  const predictButton = document.getElementById("predictButton");

  if (predictionForm) {
    predictionForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (spinner) spinner.classList.remove("d-none");
      if (predictButton) predictButton.disabled = true;

      try {
        const formData = new FormData(predictionForm);
        const payload = Object.fromEntries(formData.entries());
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await response.json();

        if (!response.ok || !data.success) {
          throw new Error(data.message || "Prediction failed");
        }

        if (resultDisplay) {
          resultDisplay.classList.remove("success", "warning", "danger");
          resultDisplay.classList.add(data.risk_level.toLowerCase());
          resultDisplay.innerHTML = `
            <span class="result-label">${data.risk_level} risk</span>
            <h3 class="mb-2">${data.probability}% probability</h3>
            <p class="mb-3">Attendance percentage: ${data.attendance_percentage}%</p>
            <div class="alert alert-light border mb-0">${data.recommendation}</div>
          `;
        }

        showToast("Prediction complete", `${data.risk_level} risk detected with ${data.probability}% probability.`, "success");
      } catch (error) {
        showToast("Prediction error", error.message, "danger");
      } finally {
        if (spinner) spinner.classList.add("d-none");
        if (predictButton) predictButton.disabled = false;
      }
    });
  }

  const downloadReportBtn = document.getElementById("downloadReportBtn");
  if (downloadReportBtn) {
    downloadReportBtn.addEventListener("click", () => {
      window.location.href = "/download-report";
    });
  }

  const studentSearch = document.getElementById("studentSearch");
  const riskFilter = document.getElementById("riskFilter");
  const applyFilterBtn = document.getElementById("applyFilterBtn");
  if (applyFilterBtn && studentSearch && riskFilter) {
    applyFilterBtn.addEventListener("click", () => {
      const params = new URLSearchParams();
      if (studentSearch.value.trim()) params.set("search", studentSearch.value.trim());
      if (riskFilter.value !== "all") params.set("risk", riskFilter.value);
      window.location.search = params.toString();
    });
  }

  const dashboardDataElement = document.getElementById("dashboard-data");
  const dashboardData = dashboardDataElement
    ? {
        weeklyLabels: safeJsonParse(dashboardDataElement.dataset.weeklyLabels, []),
        weeklyAttendance: safeJsonParse(dashboardDataElement.dataset.weeklyAttendance, []),
        weeklyRisk: safeJsonParse(dashboardDataElement.dataset.weeklyRisk, []),
      }
    : null;

  if (dashboardData && document.getElementById("weekTrendChart")) {
    const ctx = document.getElementById("weekTrendChart");
    new Chart(ctx, {
      type: "line",
      data: {
        labels: dashboardData.weeklyLabels,
        datasets: [
          {
            label: "Attendance %",
            data: dashboardData.weeklyAttendance,
            borderColor: "#3b82f6",
            backgroundColor: "rgba(59, 130, 246, 0.15)",
            tension: 0.35,
            fill: true,
          },
          {
            label: "Risk %",
            data: dashboardData.weeklyRisk,
            borderColor: "#f59e0b",
            backgroundColor: "rgba(245, 158, 11, 0.12)",
            tension: 0.35,
            fill: true,
          },
        ],
      },
      options: getChartOptions(),
    });
  }

  const analyticsDataElement = document.getElementById("analytics-data");
  const analyticsData = analyticsDataElement
    ? {
        labels: safeJsonParse(analyticsDataElement.dataset.labels, []),
        attendance: safeJsonParse(analyticsDataElement.dataset.attendance, []),
        riskValues: safeJsonParse(analyticsDataElement.dataset.riskValues, []),
        weeklyLabels: safeJsonParse(analyticsDataElement.dataset.weeklyLabels, []),
        weeklyAttendance: safeJsonParse(analyticsDataElement.dataset.weeklyAttendance, []),
        weeklyRisk: safeJsonParse(analyticsDataElement.dataset.weeklyRisk, []),
      }
    : null;

  if (analyticsData && document.getElementById("attendanceRiskChart")) {
    new Chart(document.getElementById("attendanceRiskChart"), {
      type: "bar",
      data: {
        labels: analyticsData.labels,
        datasets: [
          {
            label: "Attendance %",
            data: analyticsData.attendance,
            backgroundColor: "rgba(59, 130, 246, 0.72)",
            borderRadius: 12,
          },
        ],
      },
      options: getChartOptions(true),
    });
  }

  if (analyticsData && document.getElementById("weeklyAnalyticsChart")) {
    new Chart(document.getElementById("weeklyAnalyticsChart"), {
      type: "line",
      data: {
        labels: analyticsData.weeklyLabels,
        datasets: [
          {
            label: "Weekly attendance",
            data: analyticsData.weeklyAttendance,
            borderColor: "#1e3a8a",
            backgroundColor: "rgba(30, 58, 138, 0.15)",
            tension: 0.35,
            fill: true,
          },
          {
            label: "Weekly risk",
            data: analyticsData.weeklyRisk,
            borderColor: "#f59e0b",
            backgroundColor: "rgba(245, 158, 11, 0.12)",
            tension: 0.35,
            fill: true,
          },
        ],
      },
      options: getChartOptions(),
    });
  }
});

function getChartOptions(isBar = false) {
  return {
    responsive: true,
    plugins: {
      legend: {
        labels: {
          color: getComputedStyle(document.documentElement).getPropertyValue("--text").trim(),
        },
      },
    },
    scales: {
      x: {
        grid: { color: "rgba(148, 163, 184, 0.12)" },
        ticks: { color: getComputedStyle(document.documentElement).getPropertyValue("--muted").trim() },
      },
      y: {
        beginAtZero: true,
        grid: { color: "rgba(148, 163, 184, 0.12)" },
        ticks: { color: getComputedStyle(document.documentElement).getPropertyValue("--muted").trim() },
      },
    },
  };
}

function showToast(title, message, type) {
  const container = document.getElementById("toastContainer") || createToastContainer();
  const toast = document.createElement("div");
  toast.className = `toast align-items-center text-bg-${type} border-0 show mb-2`;
  toast.setAttribute("role", "alert");
  toast.innerHTML = `
    <div class="d-flex">
      <div class="toast-body"><strong>${title}</strong><br>${message}</div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto"></button>
    </div>
  `;
  toast.querySelector("button").addEventListener("click", () => toast.remove());
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

function safeJsonParse(value, fallback) {
  if (!value) {
    return fallback;
  }

  try {
    return JSON.parse(value);
  } catch (error) {
    return fallback;
  }
}

function createToastContainer() {
  const container = document.createElement("div");
  container.id = "toastContainer";
  container.className = "toast-container position-fixed top-0 end-0 p-3";
  document.body.appendChild(container);
  return container;
}
