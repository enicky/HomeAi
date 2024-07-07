namespace Common.Helpers;

public static class DateTimeExtensions
{
    public static string ToStartDayString(this DateTime dateTime){
        var newDateTime= new DateTime(dateTime.Year, dateTime.Month, dateTime.Day, 0,0,0,DateTimeKind.Utc);
        return newDateTime.ToString("s") + "Z";
    }
}
