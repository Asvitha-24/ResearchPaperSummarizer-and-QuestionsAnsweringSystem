"""Utility functions for text summarization and processing"""

import re


def simple_summarize(text, max_length=5000, min_length=2000):
    """Enhanced extractive summarization generating comprehensive, meaningful abstracts"""
    
    # First, clean up the text more carefully - fix OCR artifacts without breaking normal words
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Only insert space between camelCase
    text = re.sub(r'(\w)\s{2,}(?=[a-z])', r'\1', text)  # Fix broken words (internal spaces only)
    text = re.sub(r' {2,}', ' ', text)  # Multiple spaces to single
    text = re.sub(r'(\w)\n(?=[a-z])', r'\1 ', text)  # Join broken words across lines
    
    # Split by sentence endings
    sentences = re.split(r'[.!?]+', text)
    sentences = [re.sub(r' {2,}', ' ', s.strip()) for s in sentences if s.strip()]
    
    if not sentences:
        return text[:max_length] if len(text) > 0 else "No content to summarize"
    
    if len(sentences) == 1:
        return sentences[0][:max_length]
    
    # Filter out incomplete/fragmented sentences - be strict about quality
    valid_sentences = []
    for s in sentences:
        words = s.split()
        
        # Skip very short sentences
        if len(words) < 6:
            continue
        
        alpha_words = [w for w in words if any(c.isalpha() for c in w)]
        
        # Must have good alphabetic content (not metadata/numbers)
        if len(alpha_words) < len(words) * 0.65:
            continue
        
        special_ratio = sum(1 for c in s if c in '->:*[]()') / len(s)
        if special_ratio >= 0.20:
            continue
        
        valid_sentences.append(s)
    
    # If filtering removed too much, be more lenient
    if len(valid_sentences) < 4:
        valid_sentences = [s for s in sentences if len(s.split()) >= 6]
    
    if not valid_sentences:
        valid_sentences = sentences
    
    # Expanded stopwords list
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'of', 'is', 'was', 'are', 'be', 'by', 'this', 'that', 'with', 'as',
                 'from', 'into', 'up', 'about', 'which', 'who', 'it', 'its', 'they', 'them',
                 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                 'may', 'might', 'can', 'must', 'shall', 'these', 'those', 'been', 'being',
                 'such', 'no', 'not', 'only', 'just', 'also', 'all', 'each', 'every', 'we', 'us',
                 'our', 'you', 'your', 'he', 'him', 'her', 'his', 'she', 'hers', 'any', 'i', 'me',
                 'than', 'then', 'same', 'other', 'through', 'during', 'before', 'after', 'above',
                 'below', 'through', 'among', 'between', 'including', 'where', 'when', 'what', 'why', 'how'}
    
    # Calculate word frequencies from valid content - focus on content words
    all_words = re.findall(r'\w+', ' '.join(valid_sentences).lower())
    word_freq = {}
    for word in all_words:
        if len(word) > 2 and word not in stopwords:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Identify main themes (most frequent significant words) - top 20 for better coverage
    main_themes = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    theme_words = set(word for word, _ in main_themes)
    theme_importance = dict(main_themes)
    
    # Score sentences based on multiple comprehensive factors
    sentence_scores = {}
    for idx, sentence in enumerate(valid_sentences):
        sentence_words = re.findall(r'\w+', sentence.lower())
        
        # Factor 1: Theme word presence with importance weighting
        theme_score = 0
        for word in sentence_words:
            if word in theme_words:
                theme_score += theme_importance.get(word, 0)
        theme_score = theme_score / max(len(sentence_words), 1)
        
        # Factor 2: Average word frequency (how important are the words used)
        if sentence_words:
            freq_score = sum(word_freq.get(word, 0) for word in sentence_words) / len(sentence_words)
        else:
            freq_score = 0
        
        # Factor 3: Sentence position (spread across document for diverse coverage)
        normalized_pos = idx / max(len(valid_sentences) - 1, 1)
        if normalized_pos < 0.2:  # First 20%
            position_score = 0.95
        elif normalized_pos < 0.7:  # Middle 50%
            position_score = 1.0
        else:  # Last 30%
            position_score = 0.8
        
        # Factor 4: Sentence length preference (reject too short, prefer substantial sentences)
        sent_length = len(sentence_words)
        if 8 <= sent_length <= 40:
            length_score = 1.0
        elif 5 <= sent_length < 8:
            length_score = 0.5
        elif sent_length > 40:
            length_score = 0.9
        else:
            length_score = 0.2
        
        # Factor 5: Keyword density (sentences with more unique meaningful terms)
        unique_keywords = [w for w in set(sentence_words) if word_freq.get(w, 0) > 0 and len(w) > 3]
        keyword_score = len(unique_keywords) / max(len(set(sentence_words)), 1)
        
        # Factor 6: Content density - how packed with meaningful terms
        content_words = [w for w in sentence_words if w not in stopwords and len(w) > 2]
        content_density = len(content_words) / max(len(sentence_words), 1)
        
        # Combined weighted score - emphasize theme and frequency for coherent abstracts
        total_score = (theme_score * 0.32) + (freq_score * 0.28) + (keyword_score * 0.18) + (position_score * 0.12) + (length_score * 0.05) + (content_density * 0.05)
        sentence_scores[idx] = total_score
    
    # Calculate target number of sentences for comprehensive summary
    avg_sentence_length = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences) if valid_sentences else 15
    # Assume ~5 chars per word average; convert max_length (chars) to estimated sentence count
    avg_chars_per_sentence = avg_sentence_length * 5  # rough estimate
    num_sentences = max(10, int(max_length / avg_chars_per_sentence))  # Minimum 10 sentences for comprehensive abstract
    num_sentences = min(num_sentences, len(valid_sentences))
    
    # Ensure we get a diverse selection covering the document
    sorted_scores = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    selected_indices = sorted([idx for idx, _ in sorted_scores[:num_sentences]])
    summary_sentences = [valid_sentences[i] for i in selected_indices]
    
    # Build summary with proper formatting (space joining for natural flow)
    summary = ' '.join(summary_sentences)
    
    # Ensure proper punctuation
    if summary and not summary.endswith('.'):
        summary += '.'
    
    # Truncate if needed while preserving sentence structure
    if len(summary) > max_length:
        summary = summary[:max_length]
        # Remove partial sentence at end
        if '.' in summary:
            summary = summary[:summary.rfind('.')+1]
    
    # Extend if too short - fill with next best sentences
    if len(summary) < min_length and len(selected_indices) < len(valid_sentences):
        remaining = sorted([idx for idx in range(len(valid_sentences)) if idx not in selected_indices],
                          key=lambda idx: sentence_scores.get(idx, 0), reverse=True)
        for idx in remaining:
            if len(summary) < min_length:
                summary += ' ' + valid_sentences[idx]
                if not summary.endswith('.'):
                    summary += '.'
            else:
                break
    
    return summary if summary else "Unable to generate summary from content"
