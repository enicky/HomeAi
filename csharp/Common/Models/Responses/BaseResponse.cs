namespace Common.Models.Responses
{
    public abstract record BaseResponse{
        public bool Success{get;set;} = false;
    }
}