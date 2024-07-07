using System.Runtime.Serialization;

namespace Common.Models.Responses
{
    public abstract class BaseResponse{
        [DataMember]
        public bool Success{get;set;} = false;
    }
}