"""Generate the Arabic voice-alert audio file (alert_ar.mp3).

Run once to create or customize the spoken alert that plays when
drowsiness is detected. The user types their own phrase, so each
driver can personalize the warning.

Usage:
    python make_alert.py
"""

from gtts import gTTS

DEFAULT_TEXT = "انتبه! انتبه! استيقظ، أنت تشعر بالنعاس"


def main() -> None:
    print("=== إعداد صوت التنبيه ===")
    text = input("اكتب عبارة التنبيه التي تريدها: ").strip()

    if not text:
        text = DEFAULT_TEXT
        print("لم تُدخل نصاً، سنستخدم العبارة الافتراضية.")

    tts = gTTS(text=text, lang="ar")
    tts.save("alert_ar.mp3")
    print(f"تم إنشاء ملف التنبيه: alert_ar.mp3")
    print(f"العبارة: «{text}»")


if __name__ == "__main__":
    main()
