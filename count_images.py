import os

jaundice = len(os.listdir("sclera_clean/jaundice"))
normal = len(os.listdir("sclera_clean/normal"))

print("Jaundice images:", jaundice)
print("Normal images:", normal)
print("Total:", jaundice + normal)