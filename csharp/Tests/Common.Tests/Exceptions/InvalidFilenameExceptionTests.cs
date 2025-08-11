using Common.Exceptions;
using System;
using Xunit;

namespace Common.Tests.Exceptions;

public class InvalidFilenameExceptionTests
{
    [Fact]
    public void Constructor_SetsMessage()
    {
        // Arrange
        var message = "Invalid filename provided!";

        // Act
        var ex = new InvalidFilenameException(message);

        // Assert
        Assert.Equal(message, ex.Message);
    }

    [Fact]
    public void InvalidFilenameException_IsExceptionType()
    {
        // Act
        var ex = new InvalidFilenameException("msg");

        // Assert
        Assert.IsAssignableFrom<Exception>(ex);
    }
}
