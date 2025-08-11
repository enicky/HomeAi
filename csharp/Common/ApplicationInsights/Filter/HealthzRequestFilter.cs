using Microsoft.ApplicationInsights.Channel;
using Microsoft.ApplicationInsights.Extensibility;

namespace Common.ApplicationInsights.Filter;

public class HealthzRequestFilter(ITelemetryProcessor next) : ITelemetryProcessor
{
    public void Process(ITelemetry item)
    {
        if (!OKtoSend(item))
        {
            return;
        }
        next.Process(item);
    }

    private static bool OKtoSend(ITelemetry item)
    {
        var opName = item?.Context?.Operation?.Name;
        if (string.IsNullOrEmpty(opName))
        {
            // Allow telemetry if operation name is missing
            return true;
        }
        // Filter out health check requests
        if (opName == "GET /healthz" ||
            opName == "GET /health" ||
            opName == "GET /ready" ||
            opName.Contains("healthz") ||
            opName == "GET /live")
        {
            return false;
        }

        // Allow all other telemetry items
        return true;
    }
}