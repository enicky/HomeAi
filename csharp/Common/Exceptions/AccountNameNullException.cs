namespace Common.Exceptions;

public class AccountNameNullException : Exception
{
    public AccountNameNullException(string message) : base(message)
    {
    }
}
