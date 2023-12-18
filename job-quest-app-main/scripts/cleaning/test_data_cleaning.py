import re
import string
import html
import pandas as pd


def test_no_nulls(df):
    assert (
        df["job_description"].isnull().sum() == 0
    ), "There are null values in the data"
    return df


def test_lowercase(df):
    assert all(
        x.islower() or not x.isalpha() for x in df["job_description"]
    ), "Not all text is lowercase"
    return df


def test_no_html_tags(df):
    tag_re = re.compile(r"<[^>]+>")
    assert all(
        not tag_re.search(x) for x in df["job_description"]
    ), "HTML tags found in the data"
    return df


def test_html_unescape(df):
    html_entity_re = re.compile(r"&[A-Za-z0-9#]+;")
    assert all(
        not html_entity_re.search(html.unescape(x)) for x in df["job_description"]
    ), "HTML entities found in the data"
    return df


def test_no_extra_spaces(df):
    assert all(
        "  " not in x for x in df["job_description"]
    ), "Extra spaces found in the data"
    return df


def test_no_emojis(df):
    emoji_pattern = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    assert all(
        not emoji_pattern.search(x) for x in df["job_description"]
    ), "Emojis found in the data"
    return df


def test_no_punctuation(df):
    punctuation_re = re.compile("[%s]" % re.escape(string.punctuation))
    assert all(
        not punctuation_re.search(x) for x in df["job_description"]
    ), "Punctuation found in the data"
    return df


def test_no_digits(df):
    assert all(
        not any(char.isdigit() for char in x) for x in df["job_description"]
    ), "Digits found in the data"
    return df


def running_tests(df):
    # Running tests
    print("Running Tests ... ")

    # Test for Null Values
    print("Testing for Null Values...")
    df = test_no_nulls(df)
    print("✅")

    # Test for Lowercase Text
    print("Testing for Lowercase Text...")
    df = test_lowercase(df)
    print("✅")

    # Test for HTML Tags
    print("Testing for HTML Tags...")
    df = test_no_html_tags(df)
    print("✅")

    # Test for Punctuation
    print("Testing for Punctuation...")
    df = test_no_punctuation(df)
    print("✅")

    # Test for Digits
    print("Testing for Digits...")
    df = test_no_digits(df)
    print("✅")

    # Test for Emojis
    print("Testing for Emojis...")
    df = test_no_emojis(df)
    print("✅")

    # Test for Extra Spaces
    print("Testing for Extra Spaces...")
    df = test_no_extra_spaces(df)
    print("✅")

    print("All tests completed.")
    print("✅ ✅ ✅")

    return df


if __name__ == "__main__":
    # running_tests(df)
    pass
