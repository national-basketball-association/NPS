#remove existing crontab file if present
crontab -r 2> /dev/null


FILEPATH="$(pwd)"

echo "MAILTO=\"sidhantchadda@gmail.com\"" > cron
echo "00 00 * * * python /Users/sidhantchadda/NPS/"$pwd"boxScoreScraper.py" >> cron

crontab cron
rm cron
