import os
import logging
import re
import pytesseract
import numpy as np
import cv2

# For Railway deployment
WEBHOOK_URL = "web-production-1266.up.railway.app"
PORT = int(os.environ.get('PORT', 8443))

from dotenv import load_dotenv
from pytesseract import pytesseract
from PIL import Image
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TESSERACT_PATH = os.getenv("TESSERACT_PATH")

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class OCR:
    def __init__(self):
        self.path = self.find_tesseract_path()
        self.tessdata_dir = self.find_tessdata_dir()
        
        # تحسين إعداد متغيرات البيئة
        self.setup_tesseract_env()
        
        # التحقق من تثبيت اللغة العربية
        self.check_arabic_language()
    
    def setup_tesseract_env(self):
        """إعداد متغيرات بيئة Tesseract مع طبقات حماية إضافية"""
        # استخدام المسار الموجود أو المسار الافتراضي لـ Railway
        os.environ['TESSDATA_PREFIX'] = (
            self.tessdata_dir 
            or os.getenv('TESSDATA_PREFIX') 
            or '/usr/share/tesseract-ocr/5/tessdata'
        )
        
        # تأكيد مسار Tesseract
        pytesseract.tesseract_cmd = self.path or '/usr/bin/tesseract'
        
        logger.info(f"Using Tesseract path: {pytesseract.tesseract_cmd}")
        logger.info(f"Using TESSDATA_PREFIX: {os.environ['TESSDATA_PREFIX']}")

    def check_arabic_language(self):
        """التحقق من وجود اللغة العربية"""
        try:
            langs = pytesseract.get_languages(config='')
            logger.info(f"Available languages: {langs}")
            if 'ara' not in langs:
                logger.error("Arabic language pack not installed!")
        except Exception as e:
            logger.error(f"Error checking Tesseract languages: {e}")

    def find_tesseract_path(self):
        """البحث عن مسار Tesseract مع إضافة مسارات Railway"""
        paths = [
            '/usr/bin/tesseract',  # المسار الأساسي على Railway/Linux
            '/usr/local/bin/tesseract',
            '/app/bin/tesseract',  # مسار إضافي للاحتياط
            'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',  
            'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
        ]
        
        # البحث في المسارات المحددة
        for path in paths:
            if path and os.path.exists(path):
                return path
        
        # البحث في PATH النظام
        from shutil import which
        path = which('tesseract')
        if path:
            return path
            
        raise Exception("Tesseract not found. Please install Tesseract-OCR")

    def find_tessdata_dir(self):
        """البحث عن مجلد tessdata مع إضافة مسارات Railway"""
        dirs = [
            '/usr/share/tesseract-ocr/5/tessdata',  # المسار الشائع على Railway
            '/usr/share/tesseract-ocr/4.00/tessdata',
            '/usr/local/share/tessdata',
            '/app/tessdata',  # مسار مخصص إذا أضفت الملفات يدوياً
            'C:\\Program Files\\Tesseract-OCR\\tessdata'
        ]
        
        # إضافة مسار من متغير البيئة إذا كان موجوداً
        env_path = os.getenv('TESSDATA_PREFIX')
        if env_path:
            dirs.insert(0, env_path)
            
        for dir_path in dirs:
            if dir_path and os.path.exists(dir_path):
                # التحقق من وجود ملف اللغة العربية
                arabic_path = os.path.join(dir_path, 'ara.traineddata')
                if os.path.exists(arabic_path):
                    return dir_path
                else:
                    logger.warning(f"Found tessdata dir but missing Arabic language: {dir_path}")
        
        logger.error("No valid tessdata directory found with Arabic language pack")
        return None
        
def enhance_arabic_text(image_path):
    """
    دالة متقدمة لتحسين جودة الصور للنصوص العربية
    تحسينات رئيسية:
    - معالجة متعددة المراحل
    - توازن بين الحفاظ على التفاصيل وإزالة التشويش
    - تحسين خاص للخطوط العربية الدقيقة
    """
    try:
        # 1. تحميل الصورة مع الاحتفاظ بالقنوات اللونية للاستفادة منها لاحقًا
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("تعذر تحميل الصورة من المسار المحدد")
        
        # 2. تحويل إلى تدرج الرمادي مع الاحتفاظ بنسخة ملونة
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 3. معالجة مسبقة باستخدام CLAHE لتحسين التباين المحلي
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 4. تكبير الصورة مع الحفاظ على الحروف الدقيقة
        height, width = enhanced.shape
        if width < 1000:
            enhanced = cv2.resize(enhanced, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
            
        # 5. إزالة التشويش مع الحفاظ على حواف النص
        denoised = cv2.fastNlMeansDenoising(enhanced, None, h=15, templateWindowSize=7, searchWindowSize=21)
        
        # 6. Adaptive Thresholding متقدم مع ضبط ديناميكي
        cv2.GaussianBlur(enhanced, (5,5), 0)
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )        
        # 7. تحسين الحروف المتقطعة باستخدام التشكل المورفولوجي
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 8. تحسين النهاية للحصول على أفضل نتائج OCR
        final_image = cv2.bitwise_not(morph)
        
        # 9. ضبط نهائي للسطوع والتباين
        final_image = cv2.convertScaleAbs(final_image, alpha=1.5, beta=0)

        cv2.imwrite("enhanced_output.png", final_image)
        
        return Image.fromarray(final_image)
    
    except Exception as e:
        print(f"حدث خطأ في معالجة الصورة: {str(e)}")
        # استعادة الصورة الأصلية في حال حدوث خطأ
        return Image.open(image_path)
             
# إنشاء كائن OCR كمتحكم عام
ocr = OCR()

async def handle_image(update: Update, context: CallbackContext):
    if not update.message or not update.message.photo:
        return

    try:
        file = await update.message.photo[-1].get_file()
        file_path = f'temp_{file.file_id}.jpg'
        
        await file.download_to_drive(file_path)
        
        if not os.path.exists(file_path):
            await update.message.reply_text("فشل في تحميل الصورة.")
            return

        # استخراج النص باللغة الإنجليزية لاستخراج رقم الآيبان
        text_eng = ocr.extract(file_path, lang="eng")
        iban_match = re.search(r'\b[A-Z]{2}\d{2} ?(?:\d{4} ?){3,7}\d{1,4}\b', text_eng)

        iban = iban_match.group(0).replace(" ", "") if iban_match else "لم يتم العثور على رقم IBAN"

        # استخراج النص باللغة العربية
        text_ar = pytesseract.image_to_string(file_path, lang="ara", config="--psm 6")

        # تسجيل النص العربي المستخرج
        logger.info(f"📜 النص العربي المستخرج قبل التنظيف:\n{text_ar}")

        # قائمة بالكلمات التي نريد استبعادها
        excluded_words = ["بطاقة", "حساب", "جاري", "رقم", "العميل", "ايبان", "كود", "سويفت", "الشعلة", "جدة", "اوبامععم", "امع"]

        # تقسيم النص إلى سطور
        lines = text_ar.split("\n")

        # تصفية السطور التي تحتوي على كلمات محظورة
        filtered_lines = [line for line in lines if not any(word in line for word in excluded_words)]

        # إعادة تجميع النص بعد تصفية السطور
        clean_text = " ".join(filtered_lines)

        # تنظيف النص من الأرقام والرموز مع الحفاظ على الأحرف العربية والمسافات فقط
        clean_text = re.sub(r'[^أ-ي\s]', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # إزالة المسافات الزائدة

        # البحث عن الأسماء (كلمتين فأكثر) بعد إزالة الكلمات غير المطلوبة
        possible_names = re.findall(r'\b[\u0600-\u06FF]{3,}(?:\s[\u0600-\u06FF]{3,}){1,5}\b', clean_text)

        # تصفية الأسماء بحيث لا تحتوي على كلمات غير متعلقة بالأسماء
        filtered_names = [" ".join([word for word in name.split() if word not in excluded_words]) for name in possible_names]

        # إزالة القيم الفارغة بعد الفلترة
        filtered_names = [name for name in filtered_names if name.strip()]

        # اختيار أول اسم نظيف كمخرَج
        final_name = filtered_names[0] if filtered_names else "لم يتم العثور على اسم"

        logger.info(f"👤 الأسماء المستخرجة: {final_name}")

        # اختيار أول اسم صالح إذا وجد
        name = final_name

        context.user_data["copied_name"] = name
        context.user_data["copied_iban"] = iban
                
        # إنشاء الأزرار لنسخ الاسم والآيبان
        keyboard = [
            [InlineKeyboardButton("📋 نسخ الاسم", callback_data="copy_name")],
            [InlineKeyboardButton("📋 نسخ الايبان", callback_data="copy_iban")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # إرسال النتيجة مع الأزرار
        await update.message.reply_text(
            f"📄 **تم استخراج البيانات من الصورة:**\n\n"
            f"👤 **NAME:** `{name}`\n\n"
            f"🏦 **IBAN:** `{iban}`\n",  # عرض البيانات بتنسيق الكود
            parse_mode="Markdown",
            reply_markup=reply_markup
        )

    except Exception as e:
        logger.error(f"❌ خطأ أثناء معالجة الصورة: {e}")
        await update.message.reply_text("⚠️ حدث خطأ أثناء معالجة الصورة.")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

async def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query
    await query.answer()  # تجنب تعليق الطلب
    
    if query.data == "copy_name":
        text_to_copy = context.user_data.get("copied_name", "لا يوجد اسم للنسخ")
    elif query.data == "copy_iban":
        text_to_copy = context.user_data.get("copied_iban", "لا يوجد رقم IBAN للنسخ")
    else:
        text_to_copy = "خطأ غير معروف"

    await query.message.reply_text(f"تم النسخ: `{text_to_copy}`", parse_mode="Markdown")


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Welcome to the Text Extractor Bot!\n\n"
        "Send me an image containing text and I'll extract it for you."
    )

async def help_command(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "How to use this bot:\n\n"
        "1. Send me an image containing text\n"
        "2. I'll process it and return the extracted text\n\n"
        "For best results, use clear images with readable text."
    )
    
async def set_webhook(app):
    await app.bot.set_webhook(
        url=f"https://{WEBHOOK_URL}/{BOT_TOKEN}",
        allowed_updates=Update.ALL_TYPES
    )
    
def main():
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not set!")
        return

    try:
        app = ApplicationBuilder().token(BOT_TOKEN).build()
        
        # إضافة handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(MessageHandler(filters.PHOTO, handle_image))
        app.add_handler(CallbackQueryHandler(button_handler))
        
        logger.info("Bot is starting...")
        
        # حل شامل يعمل على Railway والأنظمة الأخرى
        if 'RAILWAY_STATIC_URL' in os.environ:  # إذا كان على Railway
            app.run_webhook(
                listen="0.0.0.0",
                port=PORT,
                url_path=BOT_TOKEN,
                webhook_url=f"https://{WEBHOOK_URL}/{BOT_TOKEN}",
                secret_token='WEBHOOK_SECRET'  # إضافة اختيارية للأمان

            )
        else:  # للتشغيل المحلي
            app.run_polling()
            
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")

if __name__ == '__main__':
    main()
