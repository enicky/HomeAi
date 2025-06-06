
using Microsoft.ApplicationInsights.Channel;
using Microsoft.ApplicationInsights.DataContracts;
using Microsoft.ApplicationInsights.Extensibility;

namespace Common.ApplicationInsights.Filter;

public class HangfireRequestFilter : ITelemetryProcessor
{
    private ITelemetryProcessor Next { get; set; }

    // next will point to the next TelemetryProcessor in the chain.
    public HangfireRequestFilter(ITelemetryProcessor next)
    {
        this.Next = next;
    }

    public void Process(ITelemetry item)
    {
        // To filter out an item, return without calling the next processor.
        if (!OKtoSend(item)) { return; }

        this.Next.Process(item);
    }

    // Example: replace with your own criteria.
    private static bool OKtoSend (ITelemetry item)
    {
        var dependency = item as RequestTelemetry;
        if (dependency == null) return true;
        if( dependency.Context.Operation.Name.Contains("hangfire")){
            // disable logging of hangfire requests when using UI
            return false;
        }


        return dependency.Success != true;
    }
}
