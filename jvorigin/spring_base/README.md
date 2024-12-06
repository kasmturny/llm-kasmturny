1、pom.xml管理项目依赖

2、main通过spring.xml管理其他bean组件

mapper文件——————dao的接口代码（注解@Repository）——————impl实现类（注解@Component）——————service接口（这个才是组件）

student实体类（注解@Repository）

然后就有了两个bean组件，student和service，我可以进行操作

index.jsp是前端页面，定义了action还有post和get——————通过action进入web.xml进行配置跳转——————然后再根据post和get执行对应的方法

