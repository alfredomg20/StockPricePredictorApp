from datetime import date, timedelta
import holidays

def get_next_business_days(start_date: date | str, n_days: int, country: str = 'US') -> list[date]:
    """
    Get a list of the next n business days (excluding weekends and holidays) after the given date.
    
    Args:
        start_date: The starting date, can be a date object or a string in 'YYYY-MM-DD' format
        n_days: Number of business days to return
        country: Country code for holidays (default: 'US')
        
    Returns:
        List of date objects representing the next n business days
    """
    # Get holiday calendar for the specified country
    holiday_calendar = holidays.country_holidays(country)
    
    # Convert to date object if start_date is a string
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)

    # Loop thru business days in calendar starting from the next day
    current_date = start_date + timedelta(days=1)
    business_days = []
    while len(business_days) < n_days:
        # Check if current day is a weekday and not a holiday
        is_weekday = current_date.weekday() < 5
        is_holiday = current_date in holiday_calendar
        
        if is_weekday and not is_holiday:
            business_days.append(current_date)
        
        # Move to the next day
        current_date = current_date + timedelta(days=1)
    
    return business_days

def get_last_business_day(start_date: date | str, country: str = 'US') -> date:
    """
    Get the last business day before the given start_date.
    
    Args:
        start_date: The starting date, can be a date object or a string in 'YYYY-MM-DD' format
        country: Country code for holidays (default: 'US')
        
    Returns:
        A datetime object representing the last business day before the start_date
    """
    # Get holiday calendar for the specified country
    holiday_calendar = holidays.country_holidays(country)
    
    # Convert to date object if start_date is a string
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    
    # Loop backwards to find the last business day
    current_date = start_date - timedelta(days=1)
    while True:
        # Check if current day is a weekday and not a holiday
        is_weekday = current_date.weekday() < 5
        is_holiday = current_date in holiday_calendar
        
        if is_weekday and not is_holiday:
            return current_date
        
        # Move to the previous day
        current_date = current_date - timedelta(days=1)

def is_business_day(date_to_check: date | str, country: str = 'US') -> bool:
    """
    Check if a given date is a business day (not a weekend or holiday).
    
    Args:
        date_to_check: The date to check, can be a date object or a string in 'YYYY-MM-DD' format
        country: Country code for holidays (default: 'US')
        
    Returns:
        True if the date is a business day, False otherwise
    """
    # Get holiday calendar for the specified country
    holiday_calendar = holidays.country_holidays(country)
    
    # Convert to date object if date_to_check is a string
    if isinstance(date_to_check, str):
        date_to_check = date.fromisoformat(date_to_check)
    
    # Check if the date is a weekday and not a holiday
    is_weekday = date_to_check.weekday() < 5
    is_holiday = date_to_check in holiday_calendar
    
    return is_weekday and not is_holiday

    