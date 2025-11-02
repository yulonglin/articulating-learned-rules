# Classification Rule Catalog

**Total rules in pool:** 187
**Last updated:** 2025-10-31


This catalog tracks the rule families we use to probe whether LLMs can both perform and articulate classification tasks. The unified pool merges the former core set, the vetted backlog, and archived brainstormed rules so we can monitor coverage in one place.
---

## Rule Pool Overview

### By Category

| Category | Count | Easy | Moderate | Hard | TBD |
|----------|-------|------|----------|------|-----|
| Pattern | 52 | 4 | 34 | 7 | 7 |
| Semantic | 51 | 9 | 26 | 6 | 10 |
| Statistical | 35 | 3 | 15 | 15 | 2 |
| Syntactic | 49 | 28 | 14 | 0 | 7 |

### By Implementability

| Implementation | Count | Description |
|----------------|-------|-------------|
| Programmatic | 127 | Deterministic implementation (regex, string ops, math) |
| LLM-needed | 59 | Requires semantic understanding (sentiment, intent, plans) |
| Complex | 1 | Ambiguous or difficult to implement reliably |

---

## Curated Subset for Experiments

**Purpose:** High-quality representative rules with verified articulations for empirical evaluation.

**Total curated rules:** 38
**Selection criteria:** Category-difficulty representatives with quality scores ≥0.47
**Source file:** `data/processed/list-of-rules/curated_rules_generated.jsonl`

**Note:** Curated rules have **ground truth articulations** defined in the JSONL file. These articulations are the canonical definitions used in experiments.

### Curated Rules with Ground Truth Articulations

#### Syntactic Rules

**Easy:**
- **all_caps** (0.97): The input is labeled True if all alphabetic characters are uppercase and there is at least one letter.
- **contains_multiple_exclamation_marks** (1.0): The input is labeled True if it contains two or more exclamation marks.
- **contains_consecutive_repeated_characters** (1.0): The input is labeled True if any character appears two or more times consecutively (e.g., 'oo', 'ss').
- **contains_hyphenated_word** (1.0): The input is labeled True if it contains at least one word with a hyphen connecting two parts (e.g., 'well-known').
- **word_count_less_than_5** (1.0): The input is labeled True if it contains fewer than 5 words.

**Moderate:**
- **contains_digit_pattern** (1.0): The input is labeled True if it contains a sequence of exactly three consecutive digits.
- **contains_multiple_punctuation_marks** (1.0): The input is labeled True if it contains three or more punctuation marks from the set {. , ! ? ; :}.

**Hard:**
- **Part-of-Speech Pattern** (0.90): Classify as positive if the sentence contains a pattern where a noun is immediately followed by a verb (e.g., 'dog runs'); otherwise negative.
- **PalindromeCheck** (0.90): Classify as true if the input string (ignoring spaces and case) reads the same forwards and backwards.
- **nested_quotation_depth** (0.90): Text contains quoted sections nested at least 2 levels deep (e.g., 'He said "She told me \"No\""').

#### Pattern Rules

**Easy:**
- **PresenceOfURL** (0.97): Classify as true if the input contains a URL pattern starting with 'http' or 'www'.
- **Repeated Punctuation** (0.97): Classify texts as positive if they contain a sequence of three or more identical punctuation marks (e.g., '!!!', '???'); otherwise negative.
- **starts_and_ends_same_char** (0.57): First and last non-whitespace characters are identical.
- **starts_with_vowel** (0.47): Begins with A, E, I, O, or U (case insensitive).
- **word_count_between_3_and_7** (0.47): Number of words is between 3 and 7 (inclusive).

**Moderate:**
- **alternating_case_words** (1.0): Text contains at least one word where letters alternate between uppercase and lowercase (e.g., 'aBcDeF').
- **symmetric_word_pattern** (1.0): Text contains at least one word that reads the same forwards and backwards (palindrome).
- **digit_surrounded_by_letters** (1.0): Text contains at least one digit that has a letter immediately before and after it (e.g., 'a5b').

**Hard:**
- **is_anagram_of_list** (0.50): Any word is an anagram of common words (listen, below, study).
- **rhyming_ends** (0.50): Classify the text as true if the last words of two consecutive lines rhyme; otherwise, classify as false.
- **Numeric Pattern** (0.90): Classify as positive if the input contains a date in the format 'DD/MM/YYYY' or 'Month Day, Year'; otherwise negative.
- **word_length_fibonacci** (0.90): The word lengths in the text follow the Fibonacci sequence (1, 1, 2, 3, 5, 8, etc.) for at least the first 5 words.

#### Semantic Rules

**Easy:**
- **is_adjective** (0.52): A word is classified as an adjective if it can be used to modify a noun, often appearing before it.
- **negation_presence** (0.52): Classify the text as negative if it contains any negation words (like 'not', 'no', 'never') regardless of other content.
- **first_person_perspective** (0.52): Written from first-person perspective (uses I, me, my, we, us).
- **third_person_perspective** (0.52): Written from third-person perspective (uses he, she, they, them).
- **semantic_animal_color_binding** (0.92): Text mentions at least one animal and at least one color, and explicitly binds them together in a phrase (e.g., 'red fox', 'blue whale').

**Moderate:**
- **positive_product_review** (1.0): The text expresses a positive sentiment specifically about a product or service.
- **complaint_statement** (1.0): The text explicitly states dissatisfaction or criticizes a product, service, or experience.
- **emotional_expression** (1.0): The text conveys a clear emotional state such as happiness, sadness, anger, or excitement.

**Hard:**
- **urgent_intent** (1.0): The text indicates an urgent request or call for immediate action.
- **financial_or_money_related** (1.0): The text discusses topics related to finance, money, banking, or investments.

#### Statistical Rules

**Moderate:**
- **digit_to_letter_ratio** (1.0): Text is classified as TRUE if the ratio of numeric digits to alphabetic letters is greater than 0.25, indicating substantial numerical content.
- **word_length_variance_low** (1.0): Text is classified as TRUE if the variance of word lengths (split by spaces) is less than 2.0, indicating consistent word length distribution.
- **punctuation_density_high** (1.0): Text is classified as TRUE if punctuation marks comprise more than 15% of total characters, indicating heavy punctuation usage.

**Hard:**
- **entropy_threshold_low** (1.0): Text is classified as TRUE if the Shannon entropy of character distribution is below 4.2 bits, suggesting repetitive or predictable character patterns.
- **word_length_variance_high** (1.0): Text is classified as TRUE if the variance in word lengths (excluding spaces) exceeds 8.0, indicating highly inconsistent word length distribution.
- **unique_character_ratio** (1.0): Text is classified as TRUE if the ratio of unique characters to total characters is below 0.15, suggesting limited character variety or high repetition.

---

## Rule Pool Catalog

**Complete catalog of all 187 rules** across core, vetted backlog, brainstormed, and candidate sources. See "Curated Subset for Experiments" above for the 38 high-quality rules selected for empirical evaluation.

| # | Rule Name | Category | Difficulty | Implementation | Articulation | Source |
|---|-----------|----------|------------|----------------|--------------|--------|
| 1 | all_caps | Syntactic | Easy | Programmatic | All alphabetic characters are uppercase | core |
| 2 | all_lowercase | Syntactic | Easy | Programmatic | The input is labeled True if all alphabetic characters are lowercase. | brainstorm_archive |
| 3 | alliteration_check | Pattern | Moderate | Programmatic | Classify as true if any two consecutive words start with the same letter; otherwise, classify as false. | brainstorm_archive |
| 4 | alphabetical_order | Statistical | Hard | Programmatic | First letters of consecutive words are in alphabetical order | core |
| 5 | alternating_case_pattern | Pattern | Moderate | Programmatic | String alternates between uppercase and lowercase characters starting with any case. | brainstorm_archive |
| 6 | alternating_case_word | Pattern | TBD | Programmatic | True if any word in the input alternates uppercase and lowercase letters character by character. | candidate |
| 7 | avg_word_length | Statistical | Moderate | Programmatic | Average word length is greater than 5 characters | core |
| 8 | camel_case | Syntactic | Moderate | Programmatic | The input is labeled True if it follows camelCase naming convention (no spaces, starts lowercase, subsequent words capitalized). | brainstorm_archive |
| 9 | capital_letter_ratio | Statistical | Moderate | Programmatic | More than 30% of alphabetic characters are uppercase | core |
| 10 | capitalization_effect | Statistical | Moderate | Programmatic | Classify as true if the percentage of capital letters exceeds 30%, indicating a shouty or excited tone. | brainstorm_archive |
| 11 | char_freq_vowel_ratio | Statistical | Hard | Programmatic | Classify text as 'vowel-heavy' if the ratio of vowel characters to total characters is greater than 0.4. | brainstorm_archive |
| 12 | character_ngram_entropy | Statistical | Hard | Programmatic | Text is classified as TRUE if its 4-character n-gram entropy exceeds 3.5, suggesting high information complexity. | brainstorm_archive |
| 13 | character_transition_matrix | Statistical | Hard | Programmatic | Text is classified as TRUE if its character transition probability matrix shows more than 40% predictability between adjacent characters. | brainstorm_archive |
| 14 | closing_repeats_opening | Pattern | TBD | Programmatic | True if the final sentence repeats the first word of the overall input. | candidate |
| 15 | color_word_mention | Semantic | TBD | LLM-needed | True if the text mentions a color from a predefined palette (e.g., red, blue, amber). | candidate |
| 16 | comparative_adjective | Semantic | TBD | LLM-needed | True if the text uses a comparative adjective or adverb (e.g., better, more efficient). | candidate |
| 17 | comparative_statement | Semantic | Moderate | LLM-needed | Makes a comparison between two or more things | core |
| 18 | conditional_sentence | Syntactic | Easy | Programmatic | A sentence is classified as conditional if it contains an 'if' clause indicating a condition. | brainstorm_archive |
| 19 | conditional_statement | Semantic | TBD | LLM-needed | True if the text includes an explicit conditional structure (e.g., "if X then Y"). | candidate |
| 20 | consecutive_repeated_chars | Pattern | Moderate | Programmatic | Contains at least one character that appears consecutively | core |
| 21 | contains_advice | Semantic | Moderate | LLM-needed | Contains advice or a recommendation | core |
| 22 | contains_cardinal_direction | Semantic | TBD | LLM-needed | True if the text references a cardinal direction (north, south, east, or west). | candidate |
| 23 | contains_currency_symbol | Syntactic | TBD | Programmatic | True if the text contains a currency symbol such as $ or other signs like the pound, euro, or yen symbol. | candidate |
| 24 | contains_digit | Syntactic | Easy | Programmatic | Contains at least one numeric digit | core |
| 25 | contains_email_address | Pattern | TBD | Programmatic | True if the text contains an email address pattern with '@' and a domain. | candidate |
| 26 | contains_exclamation | Syntactic | Easy | Programmatic | Contains at least one exclamation mark | core |
| 27 | contains_fraction_pattern | Pattern | TBD | Programmatic | True if the text includes a numeric fraction written with a slash (e.g., 3/4). | candidate |
| 28 | contains_hyperbole | Semantic | Moderate | LLM-needed | A sentence contains hyperbole if it uses exaggerated language to make a strong point, often including words like 'forever', 'always', or extreme quantities. | brainstorm_archive |
| 29 | contains_month_name | Semantic | TBD | LLM-needed | True if the text mentions a calendar month by name. | candidate |
| 30 | contains_negation | Semantic | Easy | LLM-needed | Contains a negation word or phrase | core |
| 31 | contains_number | Syntactic | Easy | Programmatic | The input is labeled True if it contains at least one numeric digit. | brainstorm_archive |
| 32 | contains_palindrome_word | Pattern | TBD | Programmatic | True if at least one word reads the same forward and backward and has three or more letters. | candidate |
| 33 | contains_parentheses | Syntactic | Easy | Programmatic | Contains at least one opening or closing parenthesis | core |
| 34 | contains_punctuation | Syntactic | Easy | Programmatic | The input is labeled True if it contains at least one punctuation mark. | brainstorm_archive |
| 35 | contains_question_mark | Syntactic | Easy | Programmatic | The input is labeled True if it contains at least one question mark ('?'). | brainstorm_archive |
| 36 | contains_repeated_word | Pattern | Moderate | Programmatic | Contains a word that appears more than once | core |
| 37 | contains_special_character | Syntactic | Easy | Programmatic | Contains at least one special character | core |
| 38 | contains_substring | Pattern | Moderate | Programmatic | Classify as true if the string contains the substring 'test'. | brainstorm_archive |
| 39 | contains_three_letter_word | Pattern | Moderate | Programmatic | Contains at least one word with exactly three letters | core |
| 40 | contains_time_24h | Pattern | TBD | Programmatic | True if the text includes a 24-hour timestamp in HH:MM format. | candidate |
| 41 | contains_url | Pattern | TBD | Programmatic | True if the text includes a URL beginning with http:// or https://. | candidate |
| 42 | digit_and_word_number_mix | Statistical | TBD | Programmatic | True if the text contains both numeric digits and a spelled-out number word. | candidate |
| 43 | digit_sequence | Pattern | Moderate | Programmatic | Classify as true if the string contains a sequence of three consecutive digits. | brainstorm_archive |
| 44 | digit_to_char_ratio | Statistical | Moderate | Programmatic | Digits make up more than 20% of the total characters | core |
| 45 | double_vowels | Pattern | Moderate | Programmatic | Classify as true if the string contains two consecutive vowels (aa, ee, ii, oo, uu). | brainstorm_archive |
| 46 | email_format | Pattern | Moderate | Programmatic | Classify as true if the string contains a valid email format (local-part@domain). | brainstorm_archive |
| 47 | email_like_pattern | Syntactic | Moderate | Programmatic | The input is labeled True if it contains an @ symbol and a dot, resembling an email address. | brainstorm_archive |
| 48 | emoji_presence | Syntactic | Easy | Programmatic | Classify the text as positive if it contains at least one emoji and does not contain any negative words. | brainstorm_archive |
| 49 | ends_with_period | Syntactic | Easy | Programmatic | Ends with a period ('.') | core |
| 50 | ends_with_question_mark | Syntactic | Easy | Programmatic | The input is labeled True if it ends with a question mark. | brainstorm_archive |
| 51 | even_length | Pattern | Moderate | Programmatic | Classify as true if the string has an even number of characters. | brainstorm_archive |
| 52 | exactly_two_sentences | Syntactic | TBD | Programmatic | True if the input contains exactly two sentences. | candidate |
| 53 | exclamatory_detect | Syntactic | Easy | Programmatic | A sentence is classified as exclamatory if it expresses strong emotion and ends with an exclamation mark. | brainstorm_archive |
| 54 | future_tense_check | Syntactic | Moderate | Programmatic | A sentence is classified as future tense if it contains the auxiliary verb 'will' or phrases indicating future time. | brainstorm_archive |
| 55 | hexadecimal | Pattern | Moderate | Programmatic | Classify as true if the string contains a valid hexadecimal number (0-9, A-F). | brainstorm_archive |
| 56 | hyphenated_compound | Syntactic | TBD | Programmatic | True if the text contains a hyphenated compound with letters on both sides of the hyphen. | candidate |
| 57 | imperative_detect | Syntactic | Easy | Programmatic | A sentence is classified as an imperative if it begins with a verb and gives a command without a subject. | brainstorm_archive |
| 58 | imperative_form | Semantic | Moderate | LLM-needed | Phrased as a command or request | core |
| 59 | increasing_numeric_sequence | Pattern | Moderate | Programmatic | Numeric digits in the string are strictly increasing from left to right. | brainstorm_archive |
| 60 | intent_inform | Semantic | Hard | LLM-needed | Classify texts as informative if they provide facts or explanations about a topic. | brainstorm_archive |
| 61 | intent_purchase | Semantic | Hard | LLM-needed | Classify texts as indicating intent to purchase if they mention buying or acquiring products. | brainstorm_archive |
| 62 | intent_request | Semantic | Hard | LLM-needed | Classify texts as requesting information if they ask questions or seek clarification. | brainstorm_archive |
| 63 | is_adjective | Semantic | Easy | LLM-needed | A word is classified as an adjective if it can be used to modify a noun, often appearing before it. | brainstorm_archive |
| 64 | is_anagram_of_list | Pattern | Hard | Programmatic | Any word is an anagram of common words (listen, below, study) | core |
| 65 | is_homophone_of_fair | Semantic | Moderate | LLM-needed | A word is classified as a homophone of 'fair' if it sounds the same but has a different meaning, such as 'fare'. | brainstorm_archive |
| 66 | is_noun | Syntactic | Easy | Programmatic | A word is classified as a noun if it can be preceded by a determiner (e.g., 'a', 'the'). | brainstorm_archive |
| 67 | is_plural | Pattern | Moderate | Programmatic | A word is classified as plural if it ends with the letter 's' but is not a verb in present tense. | brainstorm_archive |
| 68 | is_synonym_of_happy | Semantic | Moderate | LLM-needed | A word is classified as a synonym of 'happy' if it has a similar meaning, such as joy or gladness. | brainstorm_archive |
| 69 | is_verb | Syntactic | Easy | Programmatic | A word is classified as a verb if it can take the suffix '-ed' or '-ing'. | brainstorm_archive |
| 70 | length_greater_than_20 | Statistical | Easy | Programmatic | Total character count exceeds 20 | core |
| 71 | lengthy_sentences | Statistical | Easy | Programmatic | Classify the text as long if it contains more than 20 words; otherwise, classify as short. | brainstorm_archive |
| 72 | letter_frequency_deviation | Statistical | Hard | Programmatic | Text is classified as TRUE if its letter frequency distribution deviates more than 1.5 standard deviations from standard English letter distribution. | brainstorm_archive |
| 73 | lexical_density | Statistical | Moderate | Programmatic | Text is classified as TRUE if the ratio of unique words to total words exceeds 0.6, indicating rich vocabulary. | brainstorm_archive |
| 74 | longest_word_length | Statistical | Moderate | Programmatic | Classify text as 'advanced' if the longest word is greater than 12 characters. | brainstorm_archive |
| 75 | metaphor_detect | Semantic | Hard | LLM-needed | A sentence contains a metaphor if it compares two unlike things directly without using 'like' or 'as'. | brainstorm_archive |
| 76 | mixed_case | Syntactic | Moderate | Programmatic | Contains both uppercase and lowercase letters | core |
| 77 | multiple_sentences | Syntactic | Easy | Programmatic | Contains more than one sentence | core |
| 78 | negation_presence | Semantic | Easy | LLM-needed | Classify the text as negative if it contains any negation words (like 'not', 'no', 'never') regardless of other content. | brainstorm_archive |
| 79 | no_spaces | Syntactic | Easy | Programmatic | Contains no spaces at all | core |
| 80 | numbers_sum_to_ten | Statistical | TBD | Programmatic | True if the digits appearing in the text sum exactly to ten. | candidate |
| 81 | numeric_palindrome | Pattern | Moderate | Programmatic | Contains a numeric palindrome | core |
| 82 | palindrome_check | Pattern | Moderate | Programmatic | Identify if the text is a palindrome; if it is, classify as 'true', otherwise 'false'. | brainstorm_archive |
| 83 | palindrome_word | Syntactic | Moderate | Programmatic | The input is labeled True if the word reads the same backwards as forwards, ignoring case. | brainstorm_archive |
| 84 | passive_voice_detect | Syntactic | Moderate | Programmatic | A sentence is classified as passive voice if the subject is acted upon rather than performing the action, often using a form of the verb 'to be'. | brainstorm_archive |
| 85 | plural_noun_present | Semantic | TBD | LLM-needed | True if the text clearly references a plural noun concept (beyond simple proper-noun plurals). | candidate |
| 86 | polite_request | Semantic | TBD | LLM-needed | True if the text expresses a polite request using markers like "please" or "could you". | candidate |
| 87 | possessive_apostrophe | Syntactic | TBD | Programmatic | True if the text includes a possessive form that uses `'s` or `s'`. | candidate |
| 88 | prime_length_string | Pattern | Moderate | Programmatic | String length is a prime number between 2 and 17. | brainstorm_archive |
| 89 | punctuation_density | Statistical | Moderate | Programmatic | Ratio of punctuation marks to total characters exceeds 0.1 | core |
| 90 | punctuation_end | Pattern | Moderate | Programmatic | Classify as true if the string ends with a punctuation mark (.,!?) | brainstorm_archive |
| 91 | question_detect | Syntactic | Easy | Programmatic | A sentence is classified as a question if it ends with a question mark. | brainstorm_archive |
| 92 | question_form | Semantic | Easy | Complex | Phrased as a question | core |
| 93 | question_mark_count | Syntactic | Easy | Programmatic | Classify the text as a question if it contains at least one question mark; otherwise, classify as a statement. | brainstorm_archive |
| 94 | repeated_character | Syntactic | Moderate | Programmatic | The input is labeled True if any character appears at least three consecutive times. | brainstorm_archive |
| 95 | repeated_substring | Pattern | Moderate | Programmatic | String contains a substring that repeats at least twice, minimum length 2. | brainstorm_archive |
| 96 | repeated_substring_frequency | Statistical | Hard | Programmatic | Text is classified as TRUE if more than 15% of characters are part of repeated 3-5 character substrings. | brainstorm_archive |
| 97 | repeated_words | Pattern | Moderate | Programmatic | Classify as true if the string contains any word repeated consecutively. | brainstorm_archive |
| 98 | rhyming_ends | Pattern | Hard | Programmatic | Classify the text as true if the last words of two consecutive lines rhyme; otherwise, classify as false. | brainstorm_archive |
| 99 | sentence_contradiction | Semantic | TBD | LLM-needed | True if a later sentence contradicts an earlier sentence in the same input. | candidate |
| 100 | sentence_length_variance | Statistical | Hard | Programmatic | Classify text as 'varied' if the variance of sentence lengths is greater than 10 words. | brainstorm_archive |
| 101 | sentiment_negative | Semantic | Moderate | LLM-needed | Classify texts as negative if they express sadness, dissatisfaction, or disapproval. | brainstorm_archive |
| 102 | sentiment_neutral | Semantic | Moderate | LLM-needed | Classify texts as neutral if they do not express strong positive or negative sentiments. | brainstorm_archive |
| 103 | sentiment_positive | Semantic | Moderate | LLM-needed | Expresses a positive sentiment or emotion | core |
| 104 | sentiment_word_count | Semantic | Moderate | LLM-needed | Count positive and negative sentiment words; classify as positive if positive count exceeds negative count. | brainstorm_archive |
| 105 | special_character_bookends | Pattern | Moderate | Programmatic | String starts and ends with a special character, with alphanumeric content in between. | brainstorm_archive |
| 106 | starts_with_capital | Syntactic | Easy | Programmatic | The input is labeled True if the first character is an uppercase letter and the rest are lowercase. | brainstorm_archive |
| 107 | starts_with_number | Syntactic | TBD | Programmatic | True if the first non-space character in the input is a digit. | candidate |
| 108 | starts_with_vowel | Pattern | Easy | Programmatic | Begins with A, E, I, O, or U (case insensitive) | core |
| 109 | stopword_ratio | Statistical | Hard | Programmatic | More than 40% of words are common stopwords | core |
| 110 | syllable_complexity | Statistical | Hard | Programmatic | Text is classified as TRUE if average syllable complexity (measured by consonant clusters) exceeds 1.7. | brainstorm_archive |
| 111 | symmetric_character_count | Pattern | Moderate | Programmatic | String has an equal number of characters from first and last half. | brainstorm_archive |
| 112 | synonym_check | Semantic | Moderate | LLM-needed | A sentence contains a synonym if it includes a word that has the same or similar meaning as another word in the same sentence. | brainstorm_archive |
| 113 | temporal_reference | Semantic | Moderate | LLM-needed | Contains a reference to time (past, present, or future) | core |
| 114 | topic_environment | Semantic | Moderate | LLM-needed | Identify texts as related to the environment if they discuss ecological issues or nature. | brainstorm_archive |
| 115 | topic_finance | Semantic | Moderate | LLM-needed | Identify texts as related to finance if they discuss money management, investments, or economic terms. | brainstorm_archive |
| 116 | topic_health | Semantic | Moderate | LLM-needed | Text is about health, medicine, or wellness | core |
| 117 | topic_technology | Semantic | Moderate | LLM-needed | Identify texts as related to technology if they mention tech terms or innovations. | brainstorm_archive |
| 118 | trigraph_entropy | Statistical | Hard | Programmatic | Text is classified as TRUE if its three-character sequence entropy is between 2.5 and 3.5, suggesting complex linguistic structure. | brainstorm_archive |
| 119 | unique_word_count_ratio | Statistical | Moderate | Programmatic | Classify text as 'original' if the ratio of unique words to total words is greater than 0.5. | brainstorm_archive |
| 120 | unique_word_ratio | Statistical | Moderate | Programmatic | Ratio of unique words to total words is less than 0.7 | core |
| 121 | uppercase_start | Pattern | Moderate | Programmatic | Classify as true if the string starts with an uppercase letter. | brainstorm_archive |
| 122 | vowel_consonant_ratio | Statistical | Moderate | Programmatic | Ratio of vowels to consonants is greater than 0.6 | core |
| 123 | vowel_consonant_sandwich | Pattern | Moderate | Programmatic | String follows a pattern where a consonant is surrounded by vowels on both sides. | brainstorm_archive |
| 124 | vowel_heavy | Statistical | Hard | Programmatic | Classify the text as vowel-heavy if the ratio of vowels to consonants is greater than 1.5. | brainstorm_archive |
| 125 | word_count_between_3_and_7 | Pattern | Easy | Programmatic | Number of words is between 3 and 7 (inclusive) | core |
| 126 | word_count_greater_than_five | Pattern | Easy | Programmatic | Contains more than five words | core |
| 127 | word_length_distribution | Statistical | Hard | Programmatic | Classify text as 'diverse' if the standard deviation of word lengths is greater than 2. | brainstorm_archive |
| 128 | word_length_variance | Statistical | Moderate | Programmatic | Text is classified as TRUE if the variance of word lengths is between 1.5 and 3.0, indicating moderate lexical diversity. | brainstorm_archive |
| 129 | word_repetition_rate | Statistical | Hard | Programmatic | Classify text as 'repetitive' if any single word appears more than 3 times in a text. | brainstorm_archive |
| 130 | word_starts_with_same_letter | Pattern | Moderate | Programmatic | At least two consecutive words start with the same letter | core |
| 131 | zigzag_case | Pattern | Moderate | Programmatic | Characters alternate between uppercase and lowercase, starting with uppercase. | brainstorm_archive |
| 132 | contains_iso_date | Syntactic | Moderate | Programmatic | True if the text contains a date in ISO 8601 format (YYYY-MM-DD). | brainstorm_20251031 |
| 133 | increasing_numeric_sequence | Statistical | Hard | Programmatic | True if the text presents at least three numbers in strictly increasing order. | brainstorm_20251031 |
| 134 | question_answer_format | Pattern | Moderate | Programmatic | True if the text includes an explicit \"Q:\" followed by an \"A:\" response block. | brainstorm_20251031 |
| 135 | contains_palindrome_word | Pattern | Moderate | Programmatic | True if any word with four or more letters is a palindrome. | brainstorm_20251031 |
| 136 | contains_moral_judgment | Semantic | Hard | LLM-needed | True if the text makes an explicit moral or ethical judgment (e.g., labeling something \"immoral,\" \"wrong,\" or \"unethical\"). | brainstorm_20251031 |
| 137 | mentions_future_commitment | Semantic | Moderate | LLM-needed | True if the text promises or commits to a specific future action or deadline. | brainstorm_20251031 |
| 138 | balanced_brackets | Syntactic | Moderate | Programmatic | All opening brackets have matching closing brackets in correct order ((), [], {}). | brainstorm_20251031 |
| 139 | starts_and_ends_same_char | Pattern | Easy | Programmatic | First and last non-whitespace characters are identical. | brainstorm_20251031 |
| 140 | contains_acronym | Syntactic | Easy | Programmatic | Contains a word in all caps with 2-5 letters (e.g., USA, NASA, API). | brainstorm_20251031 |
| 141 | double_punctuation | Syntactic | Easy | Programmatic | Contains doubled punctuation marks (e.g., !!, ??, ...). | brainstorm_20251031 |
| 142 | title_case_format | Syntactic | Moderate | Programmatic | Text follows title case: first letter of each major word capitalized. | brainstorm_20251031 |
| 143 | contains_quoted_text | Syntactic | Easy | Programmatic | Contains text within quotation marks (single or double quotes). | brainstorm_20251031 |
| 144 | exactly_n_words | Statistical | Easy | Programmatic | Contains exactly N words (parameterized, e.g., N=10). | brainstorm_20251031 |
| 145 | digit_sum_threshold | Statistical | Moderate | Programmatic | Sum of all digit characters exceeds a threshold (e.g., sum > 20). | brainstorm_20251031 |
| 146 | character_diversity_ratio | Statistical | Moderate | Programmatic | Ratio of unique characters to total characters exceeds 0.7. | brainstorm_20251031 |
| 147 | consonant_cluster_frequency | Statistical | Hard | Programmatic | Contains 3+ consecutive consonants at least twice. | brainstorm_20251031 |
| 148 | haiku_structure | Pattern | Hard | LLM-needed | Text follows haiku structure: three lines with 5-7-5 syllable pattern. | brainstorm_20251031 |
| 149 | acrostic_pattern | Pattern | Moderate | Programmatic | First letters of each line/word spell out a word (minimum 3 letters). | brainstorm_20251031 |
| 150 | contains_dialogue | Syntactic | Moderate | LLM-needed | Text contains dialogue format with quotation marks and speaker attribution. | brainstorm_20251031 |
| 151 | list_format | Syntactic | Moderate | Programmatic | Text is formatted as a list with bullets, numbers, or line breaks. | brainstorm_20251031 |
| 152 | assonance_present | Pattern | Hard | LLM-needed | Contains assonance: repeated vowel sounds in nearby words. | brainstorm_20251031 |
| 153 | consonance_present | Pattern | Hard | LLM-needed | Contains consonance: repeated consonant sounds in nearby words. | brainstorm_20251031 |
| 154 | internal_rhyme | Pattern | Moderate | LLM-needed | Contains rhyming words within the same line (not just end rhymes). | brainstorm_20251031 |
| 155 | slant_rhyme | Pattern | Hard | LLM-needed | Contains slant rhyme: words with similar but not identical sounds. | brainstorm_20251031 |
| 156 | contains_harry_potter_character | Semantic | Moderate | LLM-needed | Mentions a character from Harry Potter series. | brainstorm_20251031 |
| 157 | contains_fictional_character | Semantic | Moderate | LLM-needed | References any well-known fictional character from books, movies, or TV. | brainstorm_20251031 |
| 158 | contains_city_name | Semantic | Easy | LLM-needed | Mentions a real city or town name. | brainstorm_20251031 |
| 159 | contains_country_name | Semantic | Easy | LLM-needed | References a country or nation. | brainstorm_20251031 |
| 160 | mentions_food_or_drink | Semantic | Easy | LLM-needed | Contains reference to food, beverage, or cuisine. | brainstorm_20251031 |
| 161 | contains_onomatopoeia | Semantic | Moderate | LLM-needed | Includes onomatopoeia words (e.g., boom, crash, meow, buzz). | brainstorm_20251031 |
| 162 | contains_idiom | Semantic | Moderate | LLM-needed | Uses an idiomatic expression or phrase. | brainstorm_20251031 |
| 163 | sarcastic_tone | Semantic | Hard | LLM-needed | Text conveys sarcasm or ironic tone. | brainstorm_20251031 |
| 164 | formal_register | Semantic | Moderate | LLM-needed | Language register is formal rather than casual/informal. | brainstorm_20251031 |
| 165 | contains_technical_jargon | Semantic | Moderate | LLM-needed | Uses domain-specific technical terminology. | brainstorm_20251031 |
| 166 | palindrome_sentence | Pattern | Hard | Programmatic | Entire sentence reads same forwards and backwards (ignoring spaces/punctuation). | brainstorm_20251031 |
| 167 | question_answer_pair | Syntactic | TBD | LLM-needed | Text contains both a question and its answer. | brainstorm_20251031 |
| 168 | contains_emoji_unicode | Syntactic | Easy | Programmatic | Contains Unicode emoji characters (not just :) style emoticons). | brainstorm_20251031 |
| 169 | snake_case | Syntactic | Moderate | Programmatic | Text follows snake_case convention (lowercase with underscores). | brainstorm_20251031 |
| 170 | kebab_case | Syntactic | Moderate | Programmatic | Text follows kebab-case convention (lowercase with hyphens). | brainstorm_20251031 |
| 171 | contains_ellipsis | Syntactic | Easy | Programmatic | Contains ellipsis (three or more consecutive periods). | brainstorm_20251031 |
| 172 | mixed_script | Syntactic | TBD | Programmatic | Contains characters from multiple writing systems (e.g., Latin + Cyrillic). | brainstorm_20251031 |
| 173 | increasing_word_lengths | Pattern | Moderate | Programmatic | Word lengths strictly increase from left to right. | brainstorm_20251031 |
| 174 | decreasing_word_lengths | Pattern | Moderate | Programmatic | Word lengths strictly decrease from left to right. | brainstorm_20251031 |
| 175 | word_length_symmetry | Pattern | Moderate | Programmatic | Word lengths form a symmetric pattern (palindromic lengths). | brainstorm_20251031 |
| 176 | contains_nested_parentheses | Pattern | Moderate | Programmatic | Contains parentheses within parentheses (nested at least 2 levels). | brainstorm_20251031 |
| 177 | repeating_word_pattern | Pattern | Moderate | Programmatic | Words follow an AB-AB or ABA pattern or similar repetition structure. | brainstorm_20251031 |
| 178 | contains_allcaps_word | Syntactic | Easy | Programmatic | Contains at least one word entirely in uppercase (2+ letters, excluding articles). | brainstorm_20251031 |
| 179 | average_syllables_per_word | Statistical | Moderate | LLM-needed | Average syllables per word exceeds a threshold (e.g., > 2.5). | brainstorm_20251031 |
| 180 | monosyllabic_words_only | Statistical | Moderate | LLM-needed | All words are monosyllabic (single syllable). | brainstorm_20251031 |
| 181 | contains_compound_word | Semantic | TBD | LLM-needed | Contains a compound word (e.g., butterfly, grandmother). | brainstorm_20251031 |
| 182 | contains_brand_name | Semantic | Moderate | LLM-needed | Mentions a commercial brand or product name. | brainstorm_20251031 |
| 183 | contains_historical_figure | Semantic | Moderate | LLM-needed | References a historical person or figure. | brainstorm_20251031 |
| 184 | contains_scientific_term | Semantic | Moderate | LLM-needed | Uses scientific terminology or concepts. | brainstorm_20251031 |
| 185 | first_person_perspective | Semantic | Easy | LLM-needed | Written from first-person perspective (uses I, me, my, we, us). | brainstorm_20251031 |
| 186 | third_person_perspective | Semantic | Easy | LLM-needed | Written from third-person perspective (uses he, she, they, them). | brainstorm_20251031 |
| 187 | contains_sensory_description | Semantic | TBD | LLM-needed | Includes sensory descriptions (sight, sound, smell, taste, touch). | brainstorm_20251031 |

---

## Rule Examples

### Easy Rules (sample)

**all_caps:**
- "HELLO WORLD" → True
- "Hello World" → False

**contains_digit:**
- "Hello 123" → True
- "Hello" → False

**ends_with_period:**
- "Hello world." → True
- "Hello world" → False

### Moderate Rules (sample)

**consecutive_repeated_chars:**
- "hello" → True (double 'l')
- "world" → False

**sentiment_positive:**
- "I love this!" → True
- "This is terrible" → False

**avg_word_length:**
- "magnificent incredible" → True (avg > 5)
- "cat dog" → False (avg < 5)

### Hard Rules (sample)

**is_anagram_of_list:**
- "silent" → True (anagram of "listen")
- "hello" → False

**stopword_ratio:**
- "the cat is on the mat" → True (>40% stopwords)
- "magnificent incredible discovery" → False

**alphabetical_order:**
- "apple banana cherry" → True
- "cherry apple banana" → False

---

## Related Files

- `data/processed/list-of-rules/curated_rules_generated.jsonl` — 38 high-quality rules selected for experiments (with examples, quality scores, and metadata)
- `experiments/datasets/` — curated datasets (200 samples per core rule)
- `src/generate_datasets.py` — dataset generation and export helper
- `/Users/yulong/scratch/articulating-learned-rules-backup/tmp/archive/brainstormed_rules_20251026_002228.jsonl` — scratchpad of rule ideas prior to catalog triage
