#remove existing crontab file if present
crontab -r 2> /dev/null

#stores filepath such that this can be run in any bash enviornment
FILEPATH="$(pwd)"

echo "MAILTO=\"sidhantchadda@gmail.com\"" > cron

#runs every day at midnight
echo "00 00 * * * python3 $FILEPATH/main.py" >> cron


#start cronjob
crontab cron

#remove temporary file
rm cron
