import React, { useEffect, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Tooltip,
  Legend,
  Filler,
} from "chart.js";
import { Line, Bar, Doughnut } from "react-chartjs-2";
import "./Dashboard.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Tooltip,
  Legend,
  Filler
);

interface DashboardProps {
  selectedYear: string;
  sheetType: string;
}

export default function Dashboard() {
  const [growthData, setGrowthData] = useState<number[]>([]);
  const [totalUsers, setTotalUsers] = useState<number>(0);
  const [dpData, setDpData] = useState<number[]>([]);
  const [revenueData, setRevenueData] = useState<number[]>([]);
  const [revenueLabels, setRevenueLabels] = useState<string[]>([]);

  // --- Fetch Growth Funnel ---
  useEffect(() => {
    const fetchGrowthData = async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/sheet-data/growth-funnel/1`);
        const apiData: Record<string, Record<string, { value: number; is_calculated: boolean }>> =
          await res.json();

        console.log("Growth API data:", apiData);

        const totalNetUsersRow = apiData["Total Net Users"];
        if (totalNetUsersRow) {
          const quarters = Object.keys(totalNetUsersRow).sort((a, b) => {
            const qa = parseInt(a.match(/Q(\d+)/)?.[1] ?? "0", 10);
            const qb = parseInt(b.match(/Q(\d+)/)?.[1] ?? "0", 10);
            return qa - qb;
          });

          const values = quarters.map((q) => totalNetUsersRow[q]?.value ?? 0);
          setGrowthData(values);
          setTotalUsers(values[values.length - 1] || 0);
        }
      } catch (err) {
        console.error("Error fetching growth funnel data:", err);
      }
    };

    fetchGrowthData();
  }, []);

  // --- Fetch DP Evaluation ---
  useEffect(() => {
    const fetchDPData = async () => {
      try {
        const res = await fetch(`http://localhost:8000/api/sheet-data/dp-evaluation/1`);
        const apiData: Record<string, Record<string, { value: number; is_calculated: boolean }>> =
          await res.json();

        console.log("DP API data:", apiData);

        const dpValuationRow = apiData["DP Valuation"];
        if (dpValuationRow) {
          const quarters = Object.keys(dpValuationRow).sort((a, b) => {
            const qa = parseInt(a.match(/Q(\d+)/)?.[1] ?? "0", 10);
            const qb = parseInt(b.match(/Q(\d+)/)?.[1] ?? "0", 10);
            return qa - qb;
          });

          const values = quarters.map((q) => dpValuationRow[q]?.value ?? 0);
          setDpData(values);
        }
      } catch (err) {
        console.error("Error fetching dp-evaluation data:", err);
      }
    };

    fetchDPData();
  }, []);
  useEffect(() => {
  const fetchRevenueData = async () => {
    try {
      const res = await fetch(`http://localhost:8000/api/sheet-data/revenue/1`);
      const apiData: Record<string, Record<string, { value: number; is_calculated: boolean }>> =
        await res.json();

      // Assuming "Total Revenue" contains quarterly revenue
      const totalRevenueRow = apiData["Total Revenue"];
      if (!totalRevenueRow) return;

      const quarters = Object.keys(totalRevenueRow).sort();
      const values = quarters.map((q) => totalRevenueRow[q]?.value ?? 0);

      setRevenueLabels(quarters);
      setRevenueData(values);
    } catch (err) {
      console.error("Error fetching total revenue data:", err);
    }
  };

  fetchRevenueData();
}, []);


  // --- Fetch Total Revenue for chart ---
  // --- Prepare Revenue Breakdown for Pie Chart ---
const [revenueBreakdownData, setRevenueBreakdownData] = useState<number[]>([]);
const [revenueBreakdownLabels, setRevenueBreakdownLabels] = useState<string[]>([]);

useEffect(() => {
  const fetchRevenueBreakdown = async () => {
    try {
      const res = await fetch(`http://localhost:8000/api/sheet-data/revenue/1`);
      const apiData: Record<string, Record<string, { value: number; is_calculated: boolean }>> =
        await res.json();

      const revenueKeys = [
        "Total Brokerage Revenue",
        "Total PMS Revenue",
        "Revenue from Subscriptions",
        "Revenue from Broking Interest",
        "Revenue from FPI",
        "Revenue from AUMs",
        "Net Insurance Income",
      ];

      const latestQuarter = Object.keys(apiData["Total Revenue"]).sort().pop(); // Get last quarter
      if (!latestQuarter) return;

      const values = revenueKeys.map((key) => apiData[key]?.[latestQuarter]?.value ?? 0);
      setRevenueBreakdownData(values);
      setRevenueBreakdownLabels(revenueKeys);
    } catch (err) {
      console.error("Error fetching revenue breakdown data:", err);
    }
  };

  fetchRevenueBreakdown();
}, []);

// --- Doughnut Chart Data ---
const revenuePieChartData = {
  labels: revenueBreakdownLabels,
  datasets: [
    {
      data: revenueBreakdownData,
      backgroundColor: [
        "#f97316", // orange
        "#2563eb", // blue
        "#22c55e", // green
        "#facc15", // yellow
        "#8b5cf6", // purple
        "#ec4899", // pink
        "#14b8a6", // teal
      ],
    },
  ],
};


  const formatNumber = (num: number) =>
    num?.toLocaleString("en-IN", { maximumFractionDigits: 0 });

  // Chart Data
  const growthChartData = {
    labels: growthData.map((_, idx) => `Q${idx + 1}`),
    datasets: [
      {
        label: "Total Customers",
        data: growthData,
        backgroundColor: "#f97316",
      },
    ],
  };

  const dpChartData = {
    labels: dpData.map((_, idx) => `Q${idx + 1}`),
    datasets: [
      {
        label: "DP Valuation",
        data: dpData,
        borderColor: "#2563eb",
        backgroundColor: "rgba(37, 99, 235, 0.3)",
        fill: true,
      },
    ],
  };

  const revenueChartData = {
    labels: revenueLabels,
    datasets: [
      {
        label: "Total Revenue",
        data: revenueData,
        borderColor: "#22c55e",
        backgroundColor: "rgba(34, 197, 94, 0.3)",
        fill: true,
        tension: 0.3,
      },
    ],
  };

  const dummyChartData = {
    labels: [],
    datasets: [
      {
        data: [],
        backgroundColor: [],
      },
    ],
  };

  return (
    <div className="dashboard">
      {/* Summary Cards */}
      <div className="summary-cards">
        <Card key="card-0" title="Book Value per Share" value={`NAN`} />
        <Card key="card-1" title="Total Users" value={formatNumber(totalUsers)} />
        <Card key="card-2" title="LTV / CAC Ratio" value={`NAN`} />
        <Card key="card-3" title="Monthly churn Rate" value={`NAN`} />
        <Card key="card-4" title="Closed Round" value={`NAN`} />
      </div>

      {/* Charts */}
      <div className="charts-grid">
        <ChartCard key="chart-0" title="Revenue Projections" value="" subText="">
          <Line
            data={revenueChartData}
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: { legend: { display: true } },
              scales: {
                y: {
                  ticks: {
                    callback: (value) => (value as number).toLocaleString("en-IN"),
                  },
                },
              },
            }}
          />
        </ChartCard>

        <ChartCard key="chart-4" title="Revenue Breakdown" value="" subText="">
  <Doughnut
    data={revenuePieChartData}
    options={{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "right" },
        tooltip: {
          callbacks: {
            label: function (tooltipItem) {
              const label = tooltipItem.label || "";
              const value = revenueBreakdownData[tooltipItem.dataIndex] || 0;
              return `${label}: ₹${value.toLocaleString("en-IN")}`;
            },
          },
        },
      },
    }}
  />
</ChartCard>


        <ChartCard key="chart-2" title="Customer Growth" value="" subText="">
          <Bar data={growthChartData} options={{ responsive: true, maintainAspectRatio: false }} />
        </ChartCard>

        <ChartCard key="chart-3" title="DP-Evaluation" value="" subText="">
          <Line data={dpChartData} options={{ responsive: true, maintainAspectRatio: false }} />
        </ChartCard>
      </div>
    </div>
  );
}

// ===== Reusable Components =====
interface CardProps {
  title: string;
  value: string | number;
}

function Card({ title, value }: CardProps) {
  return (
    <div className="summary-card">
      <div className="card-title">{title}</div>
      <div className="card-value">{value}</div>
    </div>
  );
}

interface ChartCardProps {
  title: string;
  value: string | number;
  subText: string;
  children: React.ReactNode;
}

function ChartCard({ title, value, subText, children }: ChartCardProps) {
  return (
    <div className="chart-card">
      <div className="chart-header">
        <div className="chart-title">{title}</div>
        <div className="chart-value">{value}</div>
        <div className="chart-subtext">{subText}</div>
      </div>
      <div className="chart-body">{children}</div>
    </div>
  );
}
