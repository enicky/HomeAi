using Common.Helpers;

namespace SchedulerService.Tests.Extensions;

public class DateTimeExtensionsTests
{
    [Fact]
    public void TestName()
    {
        // Given
        var x = DateTime.Now;
        var newDateTime= new DateTime(x.Year, x.Month, x.Day, 0,0,0,DateTimeKind.Utc);
        // When
        var y = x.ToStartDayString();
        // Then
        Assert.NotNull(y);
        Assert.Equal(newDateTime.ToString("s") + "Z", y);
    }
}
