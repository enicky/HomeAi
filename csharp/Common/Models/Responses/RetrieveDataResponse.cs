
using System.Runtime.Serialization;
using System.Runtime.Serialization.DataContracts;
using Common.Models.Influx;

namespace Common.Models.Responses;

[DataContract]
public class RetrieveDataResponse: BaseResponse{

    [DataMember]
    public IEnumerable<InfluxRecord> Value { get;  set; } = new List<InfluxRecord>();
    [DataMember]
    public string GeneratedFileName { get; set; } = string.Empty;
}