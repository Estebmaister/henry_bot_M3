"""
Mock LLM implementation for testing and demonstration purposes.
Replaces OpenAI client calls with predefined responses.
"""

import time
import random


class MockOpenAI:
    """Mock OpenAI client that simulates LLM responses."""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = MockChat()


class MockChat:
    """Mock chat completions."""

    def __init__(self):
        self.completions = MockCompletions()


class MockCompletions:
    """Mock completions."""

    def create(self, model: str, messages: list, temperature: float = 0.7, max_tokens: int = 500):
        """Generate mock responses based on the query content."""

        # Extract the user message
        user_message = ""
        for msg in messages:
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        # Simulate processing time
        time.sleep(random.uniform(0.5, 2.0))

        # Check if this is an evaluation prompt (look for JSON structure indicators)
        if "evaluate" in user_message.lower() and ("score" in user_message.lower() or "json" in user_message.lower()):
            # Return JSON format for evaluation responses
            return MockCompletion("""{
                "relevance": 9.0,
                "completeness": 8.5,
                "accuracy": 8.8,
                "reasoning": "The response provides comprehensive information about employee benefits, directly addressing the user's question with specific details about health insurance, retirement, paid time off, and other perks. The information is well-structured and actionable.",
                "recommendations": ["Consider adding specific enrollment deadlines", "Include information about benefits eligibility waiting periods"]
            }""")

        # Generate contextual responses based on query keywords
        elif any(word in user_message.lower() for word in ["benefit", "benefits", "entitled", "new employee"]):
            response = """As a new employee, you are entitled to a comprehensive benefits package including:

🏥 Health Insurance: Medical, dental, and vision coverage starting from day 1
💰 Retirement Plan: 401(k) with company matching up to 6%
⏰ Paid Time Off: 15 days vacation, 10 sick days, and 12 holidays annually
📚 Professional Development: $1,500 annual stipend for courses and certifications
🏠 Flexible Work Options: Hybrid work model with 3 days remote option
👶 Family Leave: 12 weeks paid parental leave for new parents
🏋️ Wellness Program: Free gym membership and mental health support

To enroll in benefits, log into the HR portal within your first 30 days of employment."""

        elif any(word in user_message.lower() for word in ["vacation", "time off", "pto", "leave"]):
            response = """To request vacation time, follow these steps:

1. 📅 Submit your request through the HR portal at least 2 weeks in advance
2. ⚖️ Check team calendar to ensure adequate coverage
3. 📝 Enter dates, reason, and emergency contact information
4. ✅ Your manager will receive notification and approve/deny within 3 business days
5. 📧 You'll receive email confirmation once approved

⚠️ Important Notes:
- Maximum 2 consecutive weeks for vacation requests
- Blackout periods apply during project deadlines (Dec 15-31, Mar 15-31)
- Emergency leave requests can be made through your direct manager
- Unused PTO rolls over up to 40 hours to the next year"""

        elif any(word in user_message.lower() for word in ["salary", "pay", "paycheck", "compensation"]):
            response = """Regarding compensation and payroll:

💳 Pay Schedule: Bi-weekly paychecks on Fridays (direct deposit required)
📈 Salary Reviews: Annual performance reviews with potential increases
💰 Bonuses: Performance-based bonuses up to 10% of annual salary
📊 Pay Transparency: Salary bands available in HR portal
⏰ Overtime: Eligible for 1.5x pay for hours over 40/week (non-exempt roles)

To view detailed pay information:
- Access the payroll portal using your employee credentials
- Download pay stubs and tax documents
- Set up direct deposit and tax withholdings
- Track accrued PTO and benefits usage"""

        else:
            response = """Thank you for your question! Based on the context, I recommend:

1. 📋 Check the employee handbook for detailed policies
2. 💬 Contact HR at hr@company.com for specific inquiries
3. 🖥️ Use the employee portal for forms and requests
4. 👥 Schedule a meeting with your manager for team-related questions
5. 📚 Review the onboarding materials in the knowledge base

For immediate assistance, call the HR help desk at ext. 1234 or submit a ticket through the service portal."""

        return MockCompletion(response)


class MockCompletion:
    """Mock completion response."""

    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


class MockChoice:
    """Mock choice in completion response."""

    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    """Mock message in choice response."""

    def __init__(self, content: str):
        self.content = content
