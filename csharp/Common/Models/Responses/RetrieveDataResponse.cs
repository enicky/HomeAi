
using Common.Models.Influx;

namespace Common.Models.Responses;

public record RetrieveDataResponse: BaseResponse{

    public IEnumerable<InfluxRecord> Value { get;  set; } = new List<InfluxRecord>();
    public string GeneratedFileName { get; set; } = string.Empty;
}