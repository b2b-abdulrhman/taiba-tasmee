---
title: مراجعة طيبة النشر
emoji: 📖
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# مراجعة طيبة النشر — خادم الاستماع

يسمّع الحافظ متواصلًا، ويتابع الخادم القراءة بنموذج Omnilingual ASR (من Meta) مع محاذاة قسرية لحروف النص،
وعند الخطأ الواضح تُسمَع الكلمة الصحيحة بصوت الشيخ أيمن سويد.

- صوت الشيخ في مستودع بيانات **خاص** (`AUDIO_REPO`)، ولا يُرسل منه إلا الكلمة أو البيت المطلوب.
- المتغيرات: `AUDIO_REPO` (مثل `user/taiba-audio`)، `AUDIO_FILE`، والسر `HF_TOKEN` لقراءته.
