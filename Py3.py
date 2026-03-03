import seaborn as sns
import matplotlib.pyplot as plt

# تحميل البيانات
data = sns.load_dataset('tips')

# إنشاء الرسم
plt.figure(figsize=(8,6))
sns.histplot(data=data, x='total_bill', bins=20, kde=True)

# العناوين
plt.title("Histogram of Total Bill")
plt.xlabel("Total Bill")
plt.ylabel("Frequency")

plt.show()
