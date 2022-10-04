import smtplib

def enviar_correo(subject,msg):
    user_mail = input('Email (outlook):')
    passwd = input('Password:')

    from_addr = user_mail
    to_addr = input('To address:')
    smtp_srv = "smtp-mail.outlook.com"

    message = "Subject: {}\nTo: {}\n\n{}".format(subject,to_addr,msg)

    smtp = smtplib.SMTP(smtp_srv,587)
    smtp.ehlo()
    smtp.starttls()
    smtp.ehlo()
    smtp.login(user_mail, passwd)
    smtp.sendmail(from_addr, to_addr, message)
    smtp.quit()