import smtplib

def enviar_correo(user_mail,passwd,from_addr,to_addr,subject,msg):
    smtp_srv = "smtp-mail.outlook.com"

    message = "Subject: {}\nTo: {}\n\n{}".format(subject,to_addr,msg)

    smtp = smtplib.SMTP(smtp_srv,587)
    smtp.ehlo()
    smtp.starttls()
    smtp.ehlo()
    smtp.login(user_mail, passwd)
    smtp.sendmail(from_addr, to_addr, message)
    smtp.quit()