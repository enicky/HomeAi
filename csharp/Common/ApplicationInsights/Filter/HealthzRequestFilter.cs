using Microsoft.ApplicationInsights.Channel;
using Microsoft.ApplicationInsights.Extensibility;

namespace Common.ApplicationInsights.Filter;

public class HealthzRequestFilter(ITelemetryProcessor next): ITelemetryProcessor
{
    public void Process(ITelemetry item)
    {
        if(!OKtoSend(item)) 
        { 
            return; 
        }
        next.Process(item);
    }

    private bool OKtoSend(ITelemetry item)
    {
        // Filter out health check requests
        if (item.Context.Operation.Name == "GET /healthz" ||
            item.Context.Operation.Name == "GET /health" ||
            item.Context.Operation.Name == "GET /ready" ||
            item.Context.Operation.Name.Contains("healthz") ||
            item.Context.Operation.Name == "GET /live")
        {
            return false;
        }
        {
            return false;
        }

        // Allow all other telemetry items
        return true;
    }
}